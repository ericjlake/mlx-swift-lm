import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        // We must force sorting/flattening when SSD streaming is active to properly batch
        // expert kernel dispatches dynamically over contiguous arrays.
        let isSSDStreaming = ProcessInfo.processInfo.environment["EXPERIMENTAL_SSD_STREAM"] != nil
        if isSSDStreaming { MLX.eval(indices) }
        let doSort = (indices.size >= 64) || isSSDStreaming

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        x = downProj(
            activation(xGate) * xUp,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public var tensorName: String?

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result: MLXArray
        
        // SSD Streaming: if EXPERIMENTAL_SSD_STREAM is set, route expert GEMMs through the
        // GCD dispatch_io → MTLSharedEvent → GPU hardware pipeline.
        // wShape uses logical (unquantized) dims so moe_stream_op computes output shape correctly:
        //   shape(0)=numExperts, shape(1)=outputDims, shape(2)=inputDims
        if let envPtr = getenv("EXPERIMENTAL_SSD_STREAM"), let _ = String(cString: envPtr) as String? {
           if let tensorName = self.tensorName {
               if let filename = ExpertStreamerManager.shared?.getFile(for: tensorName) {
                    let ssdPath = URL(fileURLWithPath: String(cString: envPtr)).appendingPathComponent(filename).path
                    
                    // X is already sorted and flattened by gatherSort in SwitchGLU to [N, 1, 1, inputDims]
                    // We just need to segment the sequence into contiguous chunks per expert
            
            
            MLX.eval(indices)
            if indices.size == 0 {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }
            
            let cpuIndices = indices.asArray(UInt32.self)
            
            var expertResults = [MLXArray]()
            // Broadcast a scalar zero to the required shape to avoid allocating gigabytes of zeros
            let wShape = MLX.broadcast(MLXArray(0.0), to: [self.numExperts, self.outputDims, self.inputDims])
            
            var startIdx = 0
            while startIdx < cpuIndices.count {
                let currentExpert = cpuIndices[startIdx]
                var endIdx = startIdx + 1
                while endIdx < cpuIndices.count && cpuIndices[endIdx] == currentExpert {
                    endIdx += 1
                }
                
                // Slice X across the 0th axis (batch dimension)
                let rangeX = x[startIdx ..< endIdx]
                
                let expertOutput = MLXFast.streamedGatherMM(
                    x: rangeX.asType(.float16),
                    wShape: wShape,
                    activeExpert: currentExpert,
                    safetensorsPath: ssdPath,
                    tensorName: tensorName
                )
                
                expertResults.append(expertOutput)
                
                startIdx = endIdx
            }
            
            if expertResults.isEmpty {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }
            
            let concatedResults = MLX.concatenated(expertResults, axis: 0)
            return concatedResults.asType(.float16)
               } else {
                   print("[SSD_STREAM] WARNING: filename is nil for tensorName: \(tensorName)")
               }
           } else {
               print("[SSD_STREAM] WARNING: self.tensorName is nil!")
           }
        }
        
        result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}

public class ExpertStreamerManager {
    public static var shared: ExpertStreamerManager?

    public let weightMap: [String: String]

    public init(modelDirectory: URL) {
        var map = [String: String]()
        let indexUrl = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: indexUrl),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let weightMapJson = json["weight_map"] as? [String: String] {
            map = weightMapJson
        }
        self.weightMap = map
    }

    public func getFile(for tensorName: String) -> String? {
        return weightMap[tensorName]
    }
}

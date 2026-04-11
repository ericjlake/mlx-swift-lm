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


// Shared struct for expert range tracking across projections
public struct ExpertRange {
    public let id: Int
    public let start: Int
    public let end: Int
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") public var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") public var downProj: SwitchLinear

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
        let isSSDStreaming = ExpertStreamingConfig.shared.isEnabled
        // NOTE: indices eval deferred to inside the cross-projection path below,
        // where it's merged with buffer allocation into fewer eval calls.
        let doSort = (indices.size >= 64) || isSSDStreaming

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        // ── Cross-projection batched SSD streaming path ──────────────────
        // When all 3 projections are quantized and SSD-streaming is active,
        // orchestrate buffer allocation, pread, and compute across all 3
        // projections to minimize MLX.eval() calls:
        //   - Single-token (fast path): 1 eval merges idx + buffer alloc
        //   - Prompt (large batch): 2 evals (idx, then buffers)
        //   - NO final eval — next layer's eval(idx) forces this layer
        // This reduces from 4 evals/layer (original) to 1 eval/layer.
        if isSSDStreaming,
           let qGate = gateProj as? QuantizedSwitchLinear,
           let qUp = upProj as? QuantizedSwitchLinear,
           let qDown = downProj as? QuantizedSwitchLinear,
           let gateSSD = qGate.resolveSSDInfo(),
           let upSSD = qUp.resolveSSDInfo(),
           let downSSD = qDown.resolveSSDInfo() {

            // ── EVAL REDUCTION STRATEGY ──────────────────────────────────────
            // For single-token generation (idx.size ≤ 32), we merge the sorted-
            // indices eval and buffer-allocation eval into ONE call, cutting from
            // 3 evals/layer to 1.  The final MLX.eval(x) is removed entirely:
            // the NEXT layer's SwitchGLU eval(idx) transitively forces this
            // layer's full output (including KV cache) through the lazy
            // dependency chain.  For the last layer, the generation loop's eval
            // of logits handles it.
            // ─────────────────────────────────────────────────────────────────

            if idx.size <= 32 {
                // ── FAST PATH: single-token generation ──
                // Pre-allocate max buffers (idx.size = top_k, e.g. 8) and eval
                // everything in a single call.
                let maxBuffers = idx.size
                let gateBuffers = qGate.allocateExpertBuffers(maxBuffers)
                let upBuffers = qUp.allocateExpertBuffers(maxBuffers)
                let downBuffers = qDown.allocateExpertBuffers(maxBuffers)

                // SINGLE EVAL: sorted indices + all buffer allocations.
                // Evaluating idx also transitively forces the PREVIOUS layer's
                // complete output (including KV cache) since idx depends on
                // router(prev_layer_output) via gatherSort.
                var toEval: [MLXArray] = [idx]
                toEval.append(contentsOf: gateBuffers)
                toEval.append(contentsOf: upBuffers)
                toEval.append(contentsOf: downBuffers)
                MLX.eval(toEval)

                // Handle empty indices
                if idx.size == 0 {
                    var outShape = x.shape
                    outShape[outShape.count - 1] = qDown.outputDims
                    let result = MLXArray.zeros(outShape).asType(.float16)
                    if doSort {
                        return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
                    }
                    return MLX.squeezed(result, axis: -2)
                }

                // Parse expert ranges from materialized sorted indices
                let cpuIndices = idx.asArray(UInt32.self)
                var ranges = [ExpertRange]()
                var startIdx = 0
                while startIdx < cpuIndices.count {
                    let eid = Int(cpuIndices[startIdx])
                    var endIdx = startIdx + 1
                    while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                    ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                    startIdx = endIdx
                }

                // ── CONCURRENT PREAD: all 3 projections × all experts in parallel ──
                // Dispatches up to 24 pread() syscalls simultaneously (8 experts × 3 projections),
                // pushing NVMe queue depth from QD=1 to QD=~24.
                // Raw benchmark: QD=1 = 4.5 GB/s, QD=8 = 12.3 GB/s → ~2.7× throughput.
                // preadInto is thread-safe: unique buffer per call, pread() uses explicit offset.
                let usedGate = Array(gateBuffers[0..<ranges.count])
                let usedUp = Array(upBuffers[0..<ranges.count])
                let usedDown = Array(downBuffers[0..<ranges.count])
                let totalReads = ranges.count * 3
                DispatchQueue.concurrentPerform(iterations: totalReads) { i in
                    let expertIdx = i / 3
                    let projIdx = i % 3
                    let r = ranges[expertIdx]
                    switch projIdx {
                    case 0:
                        MLXFast.preadInto(usedGate[expertIdx], safetensorsPath: gateSSD.path,
                                          tensorName: gateSSD.tensorName, expertIndex: UInt32(r.id))
                    case 1:
                        MLXFast.preadInto(usedUp[expertIdx], safetensorsPath: upSSD.path,
                                          tensorName: upSSD.tensorName, expertIndex: UInt32(r.id))
                    default:
                        MLXFast.preadInto(usedDown[expertIdx], safetensorsPath: downSSD.path,
                                          tensorName: downSSD.tensorName, expertIndex: UInt32(r.id))
                    }
                }

                // Lazy compute (no eval — next layer forces it)
                let xGate = qGate.computeExperts(x, buffers: Array(gateBuffers[0..<ranges.count]), ranges: ranges)
                let xUp = qUp.computeExperts(x, buffers: Array(upBuffers[0..<ranges.count]), ranges: ranges)
                let intermediate = activation(xGate) * xUp
                x = qDown.computeExperts(intermediate, buffers: Array(downBuffers[0..<ranges.count]), ranges: ranges)

            } else {
                // ── PROMPT PATH: larger batches ──
                // Eval indices first (needed for range count), then allocate exact buffers.
                MLX.eval(idx)

                // Handle empty indices
                if idx.size == 0 {
                    var outShape = x.shape
                    outShape[outShape.count - 1] = qDown.outputDims
                    let result = MLXArray.zeros(outShape).asType(.float16)
                    if doSort {
                        return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
                    }
                    return MLX.squeezed(result, axis: -2)
                }

                // Parse expert ranges
                let cpuIndices = idx.asArray(UInt32.self)
                var ranges = [ExpertRange]()
                var startIdx = 0
                while startIdx < cpuIndices.count {
                    let eid = Int(cpuIndices[startIdx])
                    var endIdx = startIdx + 1
                    while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                    ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                    startIdx = endIdx
                }

                // Allocate exact buffer count and eval
                let gateBuffers = qGate.allocateExpertBuffers(ranges.count)
                let upBuffers = qUp.allocateExpertBuffers(ranges.count)
                let downBuffers = qDown.allocateExpertBuffers(ranges.count)
                MLX.eval(gateBuffers + upBuffers + downBuffers)

                // Concurrent pread (same as fast path)
                let totalReads = ranges.count * 3
                DispatchQueue.concurrentPerform(iterations: totalReads) { i in
                    let expertIdx = i / 3
                    let projIdx = i % 3
                    let r = ranges[expertIdx]
                    switch projIdx {
                    case 0:
                        MLXFast.preadInto(gateBuffers[expertIdx], safetensorsPath: gateSSD.path,
                                          tensorName: gateSSD.tensorName, expertIndex: UInt32(r.id))
                    case 1:
                        MLXFast.preadInto(upBuffers[expertIdx], safetensorsPath: upSSD.path,
                                          tensorName: upSSD.tensorName, expertIndex: UInt32(r.id))
                    default:
                        MLXFast.preadInto(downBuffers[expertIdx], safetensorsPath: downSSD.path,
                                          tensorName: downSSD.tensorName, expertIndex: UInt32(r.id))
                    }
                }

                // Lazy compute (no eval — next layer forces it)
                let xGate = qGate.computeExperts(x, buffers: gateBuffers, ranges: ranges)
                let xUp = qUp.computeExperts(x, buffers: upBuffers, ranges: ranges)
                let intermediate = activation(xGate) * xUp
                x = qDown.computeExperts(intermediate, buffers: downBuffers, ranges: ranges)
            }

            if doSort {
                x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
            }
            return MLX.squeezed(x, axis: -2)
        }

        // ── Fallback: original sequential path (non-SSD or non-quantized) ──
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
    @ModuleInfo(key: "weight") public var weight: MLXArray
    @ModuleInfo(key: "bias") public var bias: MLXArray?

    public let inputDims: Int
    public let outputDims: Int
    public let numExperts: Int

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
        if ExpertStreamingConfig.shared.isEnabled {
            MLX.eval(indices)
            if indices.size == 0 {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }

            let cpuIndices = indices.asArray(UInt32.self)
            var expertResults = [MLXArray]()
            var startIdx = 0

            // macOS directNVMe: resolve the safetensors shard + tensor offset once.
            // iOS mmapPageCache: ssdInfo = nil → falls through to mmap prefault below.
            let ssdInfo: (path: String, tensorName: String)? = {
                #if os(macOS)
                guard ExpertStreamingConfig.shared.useDirectNVMe,
                      let tName = self.tensorName,
                      let filename = ExpertStreamerManager.shared?.getFile(for: tName),
                      let dir = ExpertStreamingConfig.shared.modelDirectory else { return nil }
                let path = dir.appendingPathComponent(filename).path
                return (path, tName)
                #else
                return nil  // iOS always uses mmap fallback
                #endif
            }()

            // ---- Parse expert ranges ----
            var ranges = [ExpertRange]()
            while startIdx < cpuIndices.count {
                let eid = Int(cpuIndices[startIdx])
                var endIdx = startIdx + 1
                while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                startIdx = endIdx
            }

            if let info = ssdInfo {
                // ---- Batch-allocate weight buffers (1 eval for all) ----
                var buffers = [MLXArray]()
                for _ in ranges {
                    buffers.append(MLXArray.zeros([1, self.weight.dim(1), self.weight.dim(2)]).asType(self.weight.dtype))
                }
                MLX.eval(buffers)

                // ---- Sequential pread into each fresh buffer ----
                for (i, r) in ranges.enumerated() {
                    MLXFast.preadInto(
                        buffers[i],
                        safetensorsPath: info.path,
                        tensorName: info.tensorName,
                        expertIndex: UInt32(r.id)
                    )
                }

                // ---- GPU compute for all experts ----
                for (i, r) in ranges.enumerated() {
                    let rangeX = x[r.start ..< r.end]
                    let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
                    let expertScales = self.scales[r.id ..< r.id + 1]
                    var expertBiases: MLXArray? = nil
                    if let b = self.biases { expertBiases = b[r.id ..< r.id + 1] }

                    var expertOutput = MLX.gatherQuantizedMM(
                        rangeX, buffers[i],
                        scales: expertScales, biases: expertBiases,
                        rhsIndices: expertIndices, transpose: true,
                        groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
                    )
                    if let bias = self.bias {
                        let biasSlice = bias[r.id ..< r.id + 1]
                        expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
                    }
                    let leadingShape = Array(rangeX.shape.dropLast())
                    let canonicalShape = leadingShape + [self.outputDims]
                    if expertOutput.shape != canonicalShape {
                        expertOutput = expertOutput.reshaped(canonicalShape)
                    }
                    expertResults.append(expertOutput)
                }
            } else {
                // iOS mmap fallback — original sequential path with per-expert eval
                for r in ranges {
                    let rangeX = x[r.start ..< r.end]
                    let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
                    let w = self.weight[r.id ..< r.id + 1]
                    MLX.eval(w)
                    MLXFast.prefault(w)
                    let expertScales = self.scales[r.id ..< r.id + 1]
                    var expertBiases: MLXArray? = nil
                    if let b = self.biases { expertBiases = b[r.id ..< r.id + 1] }
                    var expertOutput = MLX.gatherQuantizedMM(
                        rangeX, w,
                        scales: expertScales, biases: expertBiases,
                        rhsIndices: expertIndices, transpose: true,
                        groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
                    )
                    if let bias = self.bias {
                        let biasSlice = bias[r.id ..< r.id + 1]
                        expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
                    }
                    let leadingShape = Array(rangeX.shape.dropLast())
                    let canonicalShape = leadingShape + [self.outputDims]
                    if expertOutput.shape != canonicalShape {
                        expertOutput = expertOutput.reshaped(canonicalShape)
                    }
                    MLX.eval(expertOutput)
                    expertResults.append(expertOutput)
                }
            }

            // Batch eval all expert outputs at once (directNVMe path)
            if let _ = ssdInfo, !expertResults.isEmpty {
                MLX.eval(expertResults)
            }

            if expertResults.isEmpty {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }

            // PAPPS Heuristic: Prefetch exactly these experts so they are in cache for the N+1 token.
            if let info = ssdInfo {
                let uniqueIndices = Set(cpuIndices)
                for _ in uniqueIndices {
                    // MLXFast.pappsPrefetch(
                    //     safetensorsPath: info.path,
                    //     tensorName: info.tensorName,
                    //     expertIndex: idx
                    // )
                }
            }

            return MLX.concatenated(expertResults, axis: 0)
        }

        var result = MLX.gatherQuantizedMM(
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


    // MARK: - Cross-projection batching helpers (SSD streaming)

    /// Resolve the safetensors path and tensor name for SSD streaming.
    public func resolveSSDInfo() -> (path: String, tensorName: String)? {
        #if os(macOS)
        guard ExpertStreamingConfig.shared.useDirectNVMe,
              let tName = self.tensorName,
              let filename = ExpertStreamerManager.shared?.getFile(for: tName),
              let dir = ExpertStreamingConfig.shared.modelDirectory else { return nil }
        let path = dir.appendingPathComponent(filename).path
        return (path, tName)
        #else
        return nil
        #endif
    }

    /// Allocate zero-filled weight buffers for `count` experts (lazy, not yet eval'd).
    public func allocateExpertBuffers(_ count: Int) -> [MLXArray] {
        var buffers = [MLXArray]()
        for _ in 0..<count {
            buffers.append(MLXArray.zeros([1, self.weight.dim(1), self.weight.dim(2)]).asType(self.weight.dtype))
        }
        return buffers
    }

    /// Load expert weights from SSD into pre-allocated (eval'd) buffers.
    public func loadExpertWeights(_ buffers: [MLXArray], ranges: [ExpertRange], ssdInfo: (path: String, tensorName: String)) {
        for (i, r) in ranges.enumerated() {
            MLXFast.preadInto(
                buffers[i],
                safetensorsPath: ssdInfo.path,
                tensorName: ssdInfo.tensorName,
                expertIndex: UInt32(r.id)
            )
        }
    }

    /// Compute expert outputs using pre-loaded weight buffers. Returns LAZY result (no eval).
    public func computeExperts(_ x: MLXArray, buffers: [MLXArray], ranges: [ExpertRange]) -> MLXArray {
        var expertResults = [MLXArray]()
        for (i, r) in ranges.enumerated() {
            let rangeX = x[r.start ..< r.end]
            let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
            let expertScales = self.scales[r.id ..< r.id + 1]
            var expertBiases: MLXArray? = nil
            if let b = self.biases { expertBiases = b[r.id ..< r.id + 1] }

            var expertOutput = MLX.gatherQuantizedMM(
                rangeX, buffers[i],
                scales: expertScales, biases: expertBiases,
                rhsIndices: expertIndices, transpose: true,
                groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
            )
            if let bias = self.bias {
                let biasSlice = bias[r.id ..< r.id + 1]
                expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
            }
            let leadingShape = Array(rangeX.shape.dropLast())
            let canonicalShape = leadingShape + [self.outputDims]
            if expertOutput.shape != canonicalShape {
                expertOutput = expertOutput.reshaped(canonicalShape)
            }
            expertResults.append(expertOutput)
        }

        if expertResults.isEmpty {
            var outShape = x.shape
            outShape[outShape.count - 1] = self.outputDims
            return MLXArray.zeros(outShape).asType(.float16)
        }
        return MLX.concatenated(expertResults, axis: 0)
    }
}

public class ExpertStreamerManager {
    nonisolated(unsafe) public static var shared: ExpertStreamerManager?

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

public final class SSDStreamMetrics: @unchecked Sendable {
    public static let shared = SSDStreamMetrics()
    private var totalBytes: Int = 0
    private var totalTimeNs: UInt64 = 0
    private var readCount: Int = 0
    private var lastLogTimeNs: UInt64 = DispatchTime.now().uptimeNanoseconds
    private let lock = NSLock()
    
    public func record(bytes: Int, timeNs: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        totalBytes += bytes
        totalTimeNs += timeNs
        readCount += 1
        
        let now = DispatchTime.now().uptimeNanoseconds
        if now - lastLogTimeNs >= 1_000_000_000 {
            let count = readCount
            let bytes = totalBytes
            let ns = totalTimeNs
            
            self.readCount = 0
            self.totalBytes = 0
            self.totalTimeNs = 0
            self.lastLogTimeNs = now
            
            if count > 0 {
                // let mb = Double(bytes) / (1024.0 * 1024.0)
                // let avgMs = (Double(ns) / 1_000_000.0) / Double(count)
                // print(String(format: "[⚡️ SSD Stream] %.1f MB/s over %d chunks | Avg latency per chunk: %.6f ms", mb, count, avgMs))
                // fflush(stdout)
            }
        }
    }
}


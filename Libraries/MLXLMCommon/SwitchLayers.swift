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

// MARK: - Hot Expert Weight Cache

/// Thread-safe LRU cache for recently-loaded expert weight matrices.
///
/// MoE routing shows strong temporal locality — adjacent tokens frequently
/// route to the same experts (60-70% overlap in practice). By caching the
/// quantized weight MLXArray in unified memory, we eliminate redundant
/// pread() syscalls and MLX allocator round-trips on cache hits.
///
/// The cache key is (safetensorsPath, tensorName, expertIndex). Entries are
/// evicted in LRU order when the cache exceeds `maxEntries`.
public final class ExpertWeightCache: @unchecked Sendable {
    public static let shared = ExpertWeightCache()

    /// Maximum number of cached expert weight matrices.
    /// Each entry is typically ~200KB-2MB for 4-bit quantized experts.
    /// Default 8192 entries ≈ 1.6-16 GB depending on expert size.
    public var maxEntries: Int = 8192

    private struct CacheEntry {
        let weight: MLXArray
        var lastAccess: UInt64
    }

    private var cache = [String: CacheEntry]()
    private var accessOrder = [String]()  // oldest-first for LRU eviction
    private let lock = NSLock()

    // Metrics
    private var hits: UInt64 = 0
    private var misses: UInt64 = 0
    private var lastLogTime: UInt64 = 0

    private init() {
        lastLogTime = DispatchTime.now().uptimeNanoseconds
    }

    /// Build a unique cache key for an expert weight matrix.
    public static func key(path: String, tensorName: String, expertIndex: UInt32) -> String {
        "\(path)|\(tensorName)|\(expertIndex)"
    }

    /// Look up a cached expert weight. Returns nil on miss.
    public func get(_ key: String) -> MLXArray? {
        lock.lock()
        defer { lock.unlock() }

        guard var entry = cache[key] else {
            misses += 1
            logIfNeeded()
            return nil
        }

        hits += 1
        entry.lastAccess = DispatchTime.now().uptimeNanoseconds
        cache[key] = entry

        // Move to end of access order (most recent)
        if let idx = accessOrder.firstIndex(of: key) {
            accessOrder.remove(at: idx)
        }
        accessOrder.append(key)

        logIfNeeded()
        return entry.weight
    }

    /// Insert an expert weight into the cache. Evicts LRU entries if over capacity.
    public func put(_ key: String, weight: MLXArray) {
        lock.lock()
        defer { lock.unlock() }

        if cache[key] != nil {
            // Already cached (race between concurrent callers) — just update access
            if let idx = accessOrder.firstIndex(of: key) {
                accessOrder.remove(at: idx)
            }
            accessOrder.append(key)
            return
        }

        // Evict LRU entries if over capacity
        while cache.count >= maxEntries, !accessOrder.isEmpty {
            let evictKey = accessOrder.removeFirst()
            cache.removeValue(forKey: evictKey)
        }

        cache[key] = CacheEntry(
            weight: weight,
            lastAccess: DispatchTime.now().uptimeNanoseconds
        )
        accessOrder.append(key)
    }

    /// Clear the entire cache (e.g. on model unload).
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
        accessOrder.removeAll()
        hits = 0
        misses = 0
    }

    /// Current number of cached entries.
    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return cache.count
    }

    // Log hit rate every 10 seconds (lock must be held)
    private func logIfNeeded() {
        let now = DispatchTime.now().uptimeNanoseconds
        if now - lastLogTime >= 10_000_000_000 { // 10 seconds
            let total = hits + misses
            if total > 0 {
                let hitRate = Double(hits) / Double(total) * 100.0
                let h = hits
                let m = misses
                let c = cache.count
                // Reset window counters
                hits = 0
                misses = 0
                lastLogTime = now
                // Print outside lock would be ideal, but we're in a defer.
                // stderr to avoid interleaving with token stream.
                fputs(String(format: "[🧠 Expert Cache] %.1f%% hit rate | %llu hits, %llu misses | %d entries cached\n", hitRate, h, m, c), stderr)
                fflush(stderr)
            } else {
                lastLogTime = now
            }
        }
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
            let disableCache = ProcessInfo.processInfo.environment["DISABLE_EXPERT_CACHE"] == "1"

            if let info = ssdInfo {
                var uniqueExperts = Set<Int>()
                var prefetchIdx = 0
                while prefetchIdx < cpuIndices.count {
                    uniqueExperts.insert(Int(cpuIndices[prefetchIdx]))
                    prefetchIdx += 1
                }

                for expert in uniqueExperts {
                    let cacheKey = ExpertWeightCache.key(
                        path: info.path,
                        tensorName: info.tensorName,
                        expertIndex: UInt32(expert)
                    )
                    
                    if disableCache || ExpertWeightCache.shared.get(cacheKey) == nil {
                        MLXFast.pappsPrefetch(
                            safetensorsPath: info.path,
                            tensorName: info.tensorName,
                            expertIndex: UInt32(expert)
                        )
                    }
                }
            }

            while startIdx < cpuIndices.count {
                let currentExpert = Int(cpuIndices[startIdx])
                var endIdx = startIdx + 1
                while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == currentExpert {
                    endIdx += 1
                }

                let rangeX = x[startIdx ..< endIdx]
                let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)

                let readStart = DispatchTime.now().uptimeNanoseconds
                let expertWeight: MLXArray
                if let info = ssdInfo {
                    // ── Hot Expert Cache: check LRU before hitting SSD ──
                    let cacheKey = ExpertWeightCache.key(
                        path: info.path,
                        tensorName: info.tensorName,
                        expertIndex: UInt32(currentExpert)
                    )
                    
                    if !disableCache, let cached = ExpertWeightCache.shared.get(cacheKey) {
                        // Cache HIT — zero I/O, direct pointer reuse
                        expertWeight = cached
                    } else {
                        // Cache MISS (or disabled) — synchronous NVMe pread()
                        let w = MLXArray.zeros([1, self.weight.dim(1), self.weight.dim(2)]).asType(self.weight.dtype)
                        MLX.eval(w)
                        MLXFast.preadInto(
                            w,
                            safetensorsPath: info.path,
                            tensorName: info.tensorName,
                            expertIndex: UInt32(currentExpert)
                        )
                        expertWeight = w
                        // Insert into cache for future tokens
                        if !disableCache {
                            ExpertWeightCache.shared.put(cacheKey, weight: w)
                        }
                    }
                } else {
                    // iOS mmap / macOS mmapPageCache: page-cache backed slice.
                    let w = self.weight[currentExpert ..< currentExpert + 1]
                    MLX.eval(w)
                    MLXFast.prefault(w)
                    expertWeight = w
                }
                let readEnd = DispatchTime.now().uptimeNanoseconds
                SSDStreamMetrics.shared.record(bytes: expertWeight.nbytes, timeNs: readEnd - readStart)

                let expertScales = self.scales[currentExpert ..< currentExpert + 1]
                var expertBiases: MLXArray? = nil
                if let b = self.biases {
                    expertBiases = b[currentExpert ..< currentExpert + 1]
                }

                var expertOutput = MLX.gatherQuantizedMM(
                    rangeX,
                    expertWeight,
                    scales: expertScales,
                    biases: expertBiases,
                    rhsIndices: expertIndices,
                    transpose: true,
                    groupSize: self.groupSize,
                    bits: self.bits,
                    mode: mode,
                    sortedIndices: true
                )

                if let bias = self.bias {
                    let biasSlice = bias[currentExpert ..< currentExpert + 1]
                    expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
                }

                // Normalize to a consistent shape before concatenation.
                // gatherQuantizedMM can produce extra singleton dims (e.g. (1,1,1,1,D) vs
                // (T,1,D)) depending on whether the weight came from directNVMe or mmap,
                // causing the concatenated(expertResults, axis:0) to crash.
                // Reshape to the canonical (..., 1, outputDims) that the caller expects.
                let T = rangeX.dim(0)
                let leadingShape = Array(rangeX.shape.dropLast())  // e.g. [T, 1] or [T]
                let canonicalShape = leadingShape + [self.outputDims]
                if expertOutput.shape != canonicalShape {
                    expertOutput = expertOutput.reshaped(canonicalShape)
                }

                MLX.eval(expertOutput)
                Stream.gpu.synchronize()

                expertResults.append(expertOutput)
                startIdx = endIdx
            }

            if expertResults.isEmpty {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }

            // Hot Expert Cache: current token's experts are already cached
            // by the read loop above. No prefetch needed — the LRU cache
            // naturally retains them for reuse by subsequent tokens.

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


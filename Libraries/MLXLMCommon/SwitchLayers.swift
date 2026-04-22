import Foundation
@preconcurrency import MLX
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
public struct ExpertRange: Sendable {
    public let id: Int
    public let start: Int
    public let end: Int
}

// MARK: - SwitchGLU

public class SwitchGLU: Module, @unchecked Sendable {
    @ModuleInfo(key: "gate_proj") public var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") public var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") public var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray

    // ── Async pipeline state (SSD streaming optimization) ──
    // Persistent buffers: allocated once per layer, reused across tokens.
    // Avoids per-token buffer allocation + eval overhead.
    //
    // Hot Expert LRU cache: we allocate MAX_CACHE_SLOTS (>= top_k) buffers
    // per layer and keep experts resident across tokens. On each token we
    // only need to pread the misses — experts that aren't already cached.
    // `_slotExpert[s]` tracks which expert currently occupies slot s (nil =
    // empty). `_slotLastUsed[s]` holds the token counter at last hit/fill,
    // used for LRU eviction when all slots are full and a new expert misses.
    //
    // Memory cost scales linearly: 8 slots ≈ 5GB across 48 layers for the
    // 122B model; 16 slots ≈ 10GB. We cap below the ~13GB that was shown to
    // over-pressure the allocator in the earlier in-memory-cache experiment.
    static let MAX_CACHE_SLOTS: Int = 16

    private var _persistentGate: [MLXArray]?
    private var _persistentUp: [MLXArray]?
    private var _persistentDown: [MLXArray]?
    // Per-slot expert occupant and last-used token counter.
    private var _slotExpert: [Int?]?
    private var _slotLastUsed: [Int]?
    // Previous token's expert routing for speculative prefetch.
    private var _previousExpertIds: [Int]?
    // Per-layer token counter. Incremented each fast-path call; used by LRU.
    private var _tokenCounter: Int = 0

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
                // ── FAST PATH: single-token generation with async I/O-GPU pipeline ──
                //
                // STRATEGY: Overlap NVMe I/O with GPU compute using asyncEval,
                // plus a Hot Expert LRU cache that keeps recently-used experts
                // resident across tokens.
                //
                // Cold path (first token): Allocate MAX_CACHE_SLOTS persistent
                //   buffers per projection. Merged eval, full pread for the
                //   top-k experts, record them in slots 0..topk-1.
                //
                // Warm path (subsequent tokens):
                //   1. asyncEval(idx) kicks off GPU work (prev-layer experts,
                //      current attention + router). ~2.7 ms.
                //   2. Speculative pread during the GPU window: for each expert
                //      in `_previousExpertIds` that is NOT already cached, evict
                //      an LRU slot and issue a pread. Cached-and-already-hot
                //      experts are not re-read (free — they were already loaded).
                //   3. After GPU sync, resolve the actual routing. Hits use the
                //      cached slot with zero I/O. Misses evict the next LRU slot
                //      and pread on demand (critical path).
                //
                // Expected effect: because top_k=8 and the cache holds 16 slots,
                // hot experts that repeat across 2–3 tokens stay resident and
                // never hit the SSD a second time. Miss rate drops from ~30% to
                // ~5-15% on prose workloads.
                //
                // Memory cost: ~5GB (8 slots) → ~10GB (16 slots) across 48 layers
                // for the 122B model. Below the ~13GB ceiling that previously
                // over-pressured the allocator.

                let CACHE_SLOTS = SwitchGLU.MAX_CACHE_SLOTS

                if _persistentGate == nil {
                    // ── COLD PATH: first token, allocate persistent buffers ──
                    _persistentGate = qGate.allocateExpertBuffers(CACHE_SLOTS)
                    _persistentUp = qUp.allocateExpertBuffers(CACHE_SLOTS)
                    _persistentDown = qDown.allocateExpertBuffers(CACHE_SLOTS)
                    _slotExpert = Array(repeating: nil, count: CACHE_SLOTS)
                    _slotLastUsed = Array(repeating: 0, count: CACHE_SLOTS)
                    _tokenCounter = 0

                    // Merged eval: idx + buffer allocations (same as ssd-opt-v1)
                    var toEval: [MLXArray] = [idx]
                    toEval.append(contentsOf: _persistentGate!)
                    toEval.append(contentsOf: _persistentUp!)
                    toEval.append(contentsOf: _persistentDown!)
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

                    // Parse routing
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

                    _tokenCounter += 1

                    // Full concurrent pread — slot mapping is 1:1 with ranges.
                    let totalReads = ranges.count * 3
                    DispatchQueue.concurrentPerform(iterations: totalReads) { [ranges] i in
                        let expertIdx = i / 3
                        let projIdx = i % 3
                        let r = ranges[expertIdx]
                        switch projIdx {
                        case 0:
                            MLXFast.preadInto(self._persistentGate![expertIdx], safetensorsPath: gateSSD.path,
                                              tensorName: gateSSD.tensorName, expertIndex: UInt32(r.id))
                        case 1:
                            MLXFast.preadInto(self._persistentUp![expertIdx], safetensorsPath: upSSD.path,
                                              tensorName: upSSD.tensorName, expertIndex: UInt32(r.id))
                        default:
                            MLXFast.preadInto(self._persistentDown![expertIdx], safetensorsPath: downSSD.path,
                                              tensorName: downSSD.tensorName, expertIndex: UInt32(r.id))
                        }
                    }

                    // Record slot occupancy + LRU timestamps.
                    for (i, r) in ranges.enumerated() {
                        _slotExpert![i] = r.id
                        _slotLastUsed![i] = _tokenCounter
                    }

                    // Store routing for next token's predictions
                    _previousExpertIds = ranges.map { $0.id }

                    // Lazy compute — use slots 0..ranges.count-1
                    let usedGate = Array(_persistentGate!.prefix(ranges.count))
                    let usedUp = Array(_persistentUp!.prefix(ranges.count))
                    let usedDown = Array(_persistentDown!.prefix(ranges.count))
                    let xGate = qGate.computeExperts(x, buffers: usedGate, ranges: ranges)
                    let xUp = qUp.computeExperts(x, buffers: usedUp, ranges: ranges)
                    let intermediate = activation(xGate) * xUp
                    x = qDown.computeExperts(intermediate, buffers: usedDown, ranges: ranges)

                } else {
                    // ── WARM PATH: asyncEval + LRU-aware speculative pread ──

                    _tokenCounter += 1

                    // Start GPU work asynchronously: forces prev layer's expert
                    // compute + current layer's attention + router.
                    asyncEval(idx)

                    // Build the current slot→expert occupancy map.
                    var expertToSlot = [Int: Int]()
                    for (slot, eid) in _slotExpert!.enumerated() {
                        if let eid = eid { expertToSlot[eid] = slot }
                    }

                    // Helper: pick the slot to evict for a miss.
                    // Skips slots we've already claimed for THIS token.
                    func pickEvictionSlot(excluding claimed: Set<Int>) -> Int {
                        // Prefer empty slots.
                        for s in 0..<CACHE_SLOTS {
                            if claimed.contains(s) { continue }
                            if _slotExpert![s] == nil { return s }
                        }
                        // Else pick the least-recently-used unclaimed slot.
                        var bestSlot = -1
                        var bestTs = Int.max
                        for s in 0..<CACHE_SLOTS {
                            if claimed.contains(s) { continue }
                            if _slotLastUsed![s] < bestTs {
                                bestTs = _slotLastUsed![s]
                                bestSlot = s
                            }
                        }
                        return bestSlot
                    }

                    // ── Speculative pread: load prev-token's experts that are
                    //    NOT already cached. Cached hot experts cost us nothing.
                    var specTargets = [(expertId: Int, slot: Int)]()
                    if let prevIds = _previousExpertIds {
                        var specClaimed = Set<Int>()  // slots used by this speculation
                        for eid in prevIds {
                            if expertToSlot[eid] != nil { continue }  // already cached — skip
                            let slot = pickEvictionSlot(excluding: specClaimed)
                            if slot < 0 { break }  // all slots claimed (shouldn't happen — CACHE_SLOTS > top_k)
                            if let old = _slotExpert![slot] { expertToSlot.removeValue(forKey: old) }
                            _slotExpert![slot] = eid
                            expertToSlot[eid] = slot
                            specClaimed.insert(slot)
                            specTargets.append((eid, slot))
                        }
                    }
                    if !specTargets.isEmpty {
                        let specReads = specTargets.count * 3
                        DispatchQueue.concurrentPerform(iterations: specReads) { [specTargets] i in
                            let tIdx = i / 3
                            let proj = i % 3
                            let t = specTargets[tIdx]
                            switch proj {
                            case 0:
                                MLXFast.preadInto(self._persistentGate![t.slot], safetensorsPath: gateSSD.path,
                                                  tensorName: gateSSD.tensorName, expertIndex: UInt32(t.expertId))
                            case 1:
                                MLXFast.preadInto(self._persistentUp![t.slot], safetensorsPath: upSSD.path,
                                                  tensorName: upSSD.tensorName, expertIndex: UInt32(t.expertId))
                            default:
                                MLXFast.preadInto(self._persistentDown![t.slot], safetensorsPath: downSSD.path,
                                                  tensorName: downSSD.tensorName, expertIndex: UInt32(t.expertId))
                            }
                        }
                    }

                    // Sync on idx (blocks until GPU finishes attention + router)
                    if idx.size == 0 {
                        var outShape = x.shape
                        outShape[outShape.count - 1] = qDown.outputDims
                        let result = MLXArray.zeros(outShape).asType(.float16)
                        if doSort {
                            return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
                        }
                        return MLX.squeezed(result, axis: -2)
                    }

                    // Parse actual routing
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
                    let actualIds = ranges.map { $0.id }

                    // ── Resolve hits/misses against the cache. ──
                    var usedGate = [MLXArray]()
                    var usedUp = [MLXArray]()
                    var usedDown = [MLXArray]()
                    var claimedSlots = Set<Int>()
                    var missInfo = [(expertId: Int, slot: Int)]()

                    for r in ranges {
                        if let slot = expertToSlot[r.id], !claimedSlots.contains(slot) {
                            // HIT
                            usedGate.append(_persistentGate![slot])
                            usedUp.append(_persistentUp![slot])
                            usedDown.append(_persistentDown![slot])
                            claimedSlots.insert(slot)
                            _slotLastUsed![slot] = _tokenCounter
                        } else {
                            // MISS — evict LRU slot (not already claimed this token)
                            let slot = pickEvictionSlot(excluding: claimedSlots)
                            if slot < 0 {
                                // Should not happen; CACHE_SLOTS >= top_k.
                                fatalError("SwitchGLU cache: no slot available for expert \(r.id)")
                            }
                            if let old = _slotExpert![slot] { expertToSlot.removeValue(forKey: old) }
                            _slotExpert![slot] = r.id
                            expertToSlot[r.id] = slot
                            _slotLastUsed![slot] = _tokenCounter
                            claimedSlots.insert(slot)
                            usedGate.append(_persistentGate![slot])
                            usedUp.append(_persistentUp![slot])
                            usedDown.append(_persistentDown![slot])
                            missInfo.append((r.id, slot))
                        }
                    }

                    // Pread only misses.
                    if !missInfo.isEmpty {
                        let totalMissReads = missInfo.count * 3
                        DispatchQueue.concurrentPerform(iterations: totalMissReads) { [missInfo] i in
                            let mIdx = i / 3
                            let proj = i % 3
                            let info = missInfo[mIdx]
                            switch proj {
                            case 0:
                                MLXFast.preadInto(self._persistentGate![info.slot], safetensorsPath: gateSSD.path,
                                                  tensorName: gateSSD.tensorName, expertIndex: UInt32(info.expertId))
                            case 1:
                                MLXFast.preadInto(self._persistentUp![info.slot], safetensorsPath: upSSD.path,
                                                  tensorName: upSSD.tensorName, expertIndex: UInt32(info.expertId))
                            default:
                                MLXFast.preadInto(self._persistentDown![info.slot], safetensorsPath: downSSD.path,
                                                  tensorName: downSSD.tensorName, expertIndex: UInt32(info.expertId))
                            }
                        }
                    }

                    // Update routing for next token's predictions
                    _previousExpertIds = actualIds

                    // Lazy compute (no eval — next layer forces it)
                    let xGate = qGate.computeExperts(x, buffers: usedGate, ranges: ranges)
                    let xUp = qUp.computeExperts(x, buffers: usedUp, ranges: ranges)
                    let intermediate = activation(xGate) * xUp
                    x = qDown.computeExperts(intermediate, buffers: usedDown, ranges: ranges)
                }

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
                DispatchQueue.concurrentPerform(iterations: totalReads) { [ranges] i in
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
            if let _ = ssdInfo {
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
            _ = totalBytes
            _ = totalTimeNs
            
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


/--
Benchmarks module for Runtime Safety Kernels.

This module provides comprehensive performance benchmarking for all RSK components,
including latency measurements, throughput analysis, and performance regression detection.
-/

import RuntimeSafetyKernels.Sampler
import RuntimeSafetyKernels.Concurrency
import RuntimeSafetyKernels.Policy
import RuntimeSafetyKernels.Shape

/-- Benchmarks module -/
module RuntimeSafetyKernels.Benchmarks

/-- Benchmark result -/
structure BenchmarkResult where
  name : String
  iterations : Nat
  totalTimeMs : Float
  averageTimeMs : Float
  throughput : Float  -- operations per second
  memoryUsage : Option Float  -- MB
  deriving Repr

/-- Performance baseline -/
structure PerformanceBaseline where
  component : String
  operation : String
  targetLatencyMs : Float
  targetThroughput : Float
  maxMemoryMB : Float
  deriving Repr

/-- Default performance baselines -/
def defaultBaselines : List PerformanceBaseline := [
  ⟨"Sampler", "topK", 0.1, 10000.0, 1.0⟩,
  ⟨"Sampler", "topP", 0.15, 8000.0, 1.0⟩,
  ⟨"Sampler", "mirostat", 0.5, 2000.0, 2.0⟩,
  ⟨"Concurrency", "submit", 0.01, 100000.0, 0.1⟩,
  ⟨"Concurrency", "process", 0.05, 20000.0, 0.1⟩,
  ⟨"Policy", "guard", 0.02, 50000.0, 0.1⟩,
  ⟨"Policy", "decode", 0.03, 30000.0, 0.1⟩,
  ⟨"Shape", "create", 0.1, 10000.0, 0.5⟩,
  ⟨"Shape", "multiply", 1.0, 1000.0, 2.0⟩,
  ⟨"Shape", "transpose", 0.2, 5000.0, 0.5⟩
]

/-- Generate test data for benchmarks -/
def generateBenchmarkData : IO (Vector Float 1000 × Vector Float 1000 × TensorShape × TensorData) := do
  -- Generate logits
  let logits := Vector.generate 1000 (fun _ => Float.random)

  -- Generate tensor data
  let tensorData := Vector.generate 1000 (fun _ => Float.random)

  -- Generate tensor shape (20x50)
  let shape := TensorShape.mk #[20, 50]

  -- Generate tensor
  let tensor := TensorData.mk tensorData shape

  return (logits, tensorData, shape, tensor)

/-- Benchmark sampling algorithms -/
def benchmarkSampling (iterations : Nat := 10000) : IO (List BenchmarkResult) := do
  let (logits, _, _, _) ← generateBenchmarkData

  let samplingConfigs := [
    ("topK-40", SamplingConfig.topK ⟨40, 1.0⟩),
    ("topK-100", SamplingConfig.topK ⟨100, 1.0⟩),
    ("topP-0.9", SamplingConfig.topP ⟨0.9, 1.0⟩),
    ("topP-0.95", SamplingConfig.topP ⟨0.95, 1.0⟩),
    ("mirostat-3.0", SamplingConfig.mirostat ⟨3.0, 0.1, 100, 0.01⟩)
  ]

  let mutable results := []

  for (name, config) in samplingConfigs do
    let startTime ← IO.monoMsNow

    for _ in List.range iterations do
      let _ := sample logits config
      pure ()

    let endTime ← IO.monoMsNow
    let totalTime := endTime - startTime
    let avgTime := totalTime / iterations.toFloat
    let throughput := (iterations.toFloat * 1000.0) / totalTime

    results := results ++ [⟨name, iterations, totalTime, avgTime, throughput, none⟩]

  return results

/-- Benchmark concurrency operations -/
def benchmarkConcurrency (iterations : Nat := 10000) : IO (List BenchmarkResult) := do
  let initialState := ConcurrencyState.mk
    (Vector.mkArray 8 WorkerState.idle)
    []
    0
    0
    0

  let request := RuntimeRequest.mk 1 1 100 5000 #[1, 2, 3, 4, 5]

  -- Benchmark submit operations
  let mutable state := initialState
  let startTime ← IO.monoMsNow

  for _ in List.range iterations do
    match submitRequest state request with
    | none => pure () -- Queue full, reset
    | some newState => state := newState

  let endTime ← IO.monoMsNow
  let totalTime := endTime - startTime
  let avgTime := totalTime / iterations.toFloat
  let throughput := (iterations.toFloat * 1000.0) / totalTime

  let submitResult := ⟨"submit", iterations, totalTime, avgTime, throughput, none⟩

  -- Benchmark event processing
  let event := Event.complete 0 1
  let startTime2 ← IO.monoMsNow

  for _ in List.range iterations do
    let _ := processEvent initialState event
    pure ()

  let endTime2 ← IO.monoMsNow
  let totalTime2 := endTime2 - startTime2
  let avgTime2 := totalTime2 / iterations.toFloat
  let throughput2 := (iterations.toFloat * 1000.0) / totalTime2

  let processResult := ⟨"process", iterations, totalTime2, avgTime2, throughput2, none⟩

  return [submitResult, processResult]

/-- Benchmark policy operations -/
def benchmarkPolicy (iterations : Nat := 10000) : IO (List BenchmarkResult) := do
  let policyConfig := PolicyConfig.mk false [1, 2, 3] 100 1000 100
  let decoderState := DecoderState.mk 0 0 0 0

  -- Benchmark policy guard
  let startTime ← IO.monoMsNow

  for i in List.range iterations do
    let token := i % 1000
    let currentTime := i
    let _ := policyGuard policyConfig decoderState token currentTime
    pure ()

  let endTime ← IO.monoMsNow
  let totalTime := endTime - startTime
  let avgTime := totalTime / iterations.toFloat
  let throughput := (iterations.toFloat * 1000.0) / totalTime

  let guardResult := ⟨"guard", iterations, totalTime, avgTime, throughput, none⟩

  -- Benchmark decode operations
  let startTime2 ← IO.monoMsNow

  for i in List.range iterations do
    let token := i % 1000
    let currentTime := i
    let _ := decode policyConfig decoderState token currentTime
    pure ()

  let endTime2 ← IO.monoMsNow
  let totalTime2 := endTime2 - startTime2
  let avgTime2 := totalTime2 / iterations.toFloat
  let throughput2 := (iterations.toFloat * 1000.0) / totalTime2

  let decodeResult := ⟨"decode", iterations, totalTime2, avgTime2, throughput2, none⟩

  return [guardResult, decodeResult]

/-- Benchmark shape operations -/
def benchmarkShape (iterations : Nat := 1000) : IO (List BenchmarkResult) := do
  let (_, tensorData, shape, tensor) ← generateBenchmarkData

  let mutable results := []

  -- Benchmark tensor creation
  let startTime ← IO.monoMsNow
  for _ in List.range iterations do
    let _ := createTensor tensorData shape
    pure ()
  let endTime ← IO.monoMsNow
  let totalTime := endTime - startTime
  let avgTime := totalTime / iterations.toFloat
  let throughput := (iterations.toFloat * 1000.0) / totalTime
  results := results ++ [⟨"create", iterations, totalTime, avgTime, throughput, none⟩]

  -- Benchmark matrix multiplication
  let startTime2 ← IO.monoMsNow
  for _ in List.range iterations do
    let _ := matrixMultiply tensor tensor
    pure ()
  let endTime2 ← IO.monoMsNow
  let totalTime2 := endTime2 - startTime2
  let avgTime2 := totalTime2 / iterations.toFloat
  let throughput2 := (iterations.toFloat * 1000.0) / totalTime2
  results := results ++ [⟨"multiply", iterations, totalTime2, avgTime2, throughput2, none⟩]

  -- Benchmark transpose
  let startTime3 ← IO.monoMsNow
  for _ in List.range iterations do
    let _ := transpose tensor
    pure ()
  let endTime3 ← IO.monoMsNow
  let totalTime3 := endTime3 - startTime3
  let avgTime3 := totalTime3 / iterations.toFloat
  let throughput3 := (iterations.toFloat * 1000.0) / totalTime3
  results := results ++ [⟨"transpose", iterations, totalTime3, avgTime3, throughput3, none⟩]

  return results

/-- Benchmark integration scenarios -/
def benchmarkIntegration (iterations : Nat := 1000) : IO (List BenchmarkResult) := do
  let (logits, _, _, _) ← generateBenchmarkData

  -- End-to-end AI inference pipeline
  let startTime ← IO.monoMsNow

  for _ in List.range iterations do
    -- 1. Sample token
    let samplingConfig := SamplingConfig.topK ⟨40, 1.0⟩
    let samplingResult := sample logits samplingConfig
    let token := samplingResult.selectedToken

    -- 2. Apply policy guard
    let policyConfig := PolicyConfig.mk false [1, 2, 3] 100 1000 100
    let decoderState := DecoderState.mk 0 0 0 0
    let currentTime := 1000

    let (guardResult, newDecoderState) := policyGuard policyConfig decoderState token currentTime

    -- 3. Process through concurrency system
    let concurrencyState := ConcurrencyState.mk
      (Vector.mkArray 4 WorkerState.idle)
      []
      0
      0
      0

    let request := RuntimeRequest.mk 1 1 100 5000 #[token]

    match submitRequest concurrencyState request with
    | none => pure () -- Queue full
    | some newState =>
      let finalState := processEvent newState (Event.complete 0 request.id)
      pure ()

  let endTime ← IO.monoMsNow
  let totalTime := endTime - startTime
  let avgTime := totalTime / iterations.toFloat
  let throughput := (iterations.toFloat * 1000.0) / totalTime

  return [⟨"integration", iterations, totalTime, avgTime, throughput, none⟩]

/-- Run all benchmarks -/
def runAllBenchmarks (iterations : Nat := 10000) : IO (List (String × List BenchmarkResult)) := do
  let benchmarks := [
    ("Sampling", benchmarkSampling),
    ("Concurrency", benchmarkConcurrency),
    ("Policy", benchmarkPolicy),
    ("Shape", benchmarkShape),
    ("Integration", benchmarkIntegration)
  ]

  let mutable results := []

  for (name, benchmark) in benchmarks do
    IO.println s!"Running {name} benchmarks..."
    let result ← benchmark iterations
    results := results ++ [(name, result)]

  return results

/-- Check performance against baselines -/
def checkPerformanceBaselines (results : List (String × List BenchmarkResult)) : IO (List (String × Bool)) := do
  let mutable baselineChecks := []

  for (component, componentResults) in results do
    for result in componentResults do
      let baseline := defaultBaselines.find? (fun b => b.component == component && b.operation == result.name)

      match baseline with
      | none => pure () -- No baseline for this operation
      | some baseline =>
        let latencyOk := result.averageTimeMs <= baseline.targetLatencyMs
        let throughputOk := result.throughput >= baseline.targetThroughput

        let passed := latencyOk && throughputOk
        baselineChecks := baselineChecks ++ [⟨s!"{component}.{result.name}", passed⟩]

        if !passed then
          IO.println s!"Performance regression in {component}.{result.name}:"
          IO.println s!"  Latency: {result.averageTimeMs}ms (target: {baseline.targetLatencyMs}ms)"
          IO.println s!"  Throughput: {result.throughput} ops/s (target: {baseline.targetThroughput} ops/s)"

  return baselineChecks

/-- Generate performance report -/
def generatePerformanceReport (results : List (String × List BenchmarkResult)) : String :=
  let reportLines := results.map (fun (component, componentResults) =>
    s!"\n{component} Benchmarks:\n" ++
    (componentResults.map (fun result =>
      s!"  {result.name}: {result.averageTimeMs:.3f}ms avg, {result.throughput:.0f} ops/s"
    )).join "\n"
  )

  "Runtime Safety Kernels Performance Report\n" ++
  "==========================================\n" ++
  reportLines.join "\n"

/-- Main benchmarking entry point -/
def main (args : List String) : IO Unit := do
  let iterations := if args.contains "--performance" then 100000 else 10000
  let compareBaseline := args.contains "--compare-baseline"

  IO.println s!"Starting benchmarks with {iterations} iterations per test..."

  let results ← runAllBenchmarks iterations

  -- Print results
  IO.println (generatePerformanceReport results)

  -- Check baselines if requested
  if compareBaseline then
    IO.println "\nChecking performance baselines..."
    let baselineChecks ← checkPerformanceBaselines results

    let passed := baselineChecks.filter (fun (_, passed) => passed) |>.length
    let failed := baselineChecks.filter (fun (_, passed) => !passed) |>.length

    IO.println s!"Baseline checks: {passed} passed, {failed} failed"

    if failed > 0 then
      IO.println "✗ Performance regression detected!"
      IO.Process.exit 1
    else
      IO.println "✓ All performance baselines met!"
  else
    IO.println "✓ Benchmarks completed successfully!"

/-- Export for Lake build -/
#eval main []

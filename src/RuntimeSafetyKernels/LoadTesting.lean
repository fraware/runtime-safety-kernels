/--
Load testing module for Runtime Safety Kernels.

This module provides comprehensive load testing to validate performance requirements:
- 4,096 concurrent requests
- p99 queueing latency < 250 µs
- 500 rps throughput
- Ultra-low latency policy guarding
-/

import RuntimeSafetyKernels.Sampler
import RuntimeSafetyKernels.Concurrency
import RuntimeSafetyKernels.Policy
import RuntimeSafetyKernels.Shape

/-- Load testing module -/
module RuntimeSafetyKernels.LoadTesting

/-- Load test configuration -/
structure LoadTestConfig where
  concurrentRequests : Nat
  requestsPerSecond : Nat
  testDurationSeconds : Nat
  maxTokensPerRequest : Nat
  contextWindowSize : Nat
  targetP99LatencyMs : Float
  targetThroughput : Nat
  deriving Repr

/-- Load test result -/
structure LoadTestResult where
  totalRequests : Nat
  successfulRequests : Nat
  failedRequests : Nat
  averageLatencyMs : Float
  p50LatencyMs : Float
  p95LatencyMs : Float
  p99LatencyMs : Float
  p999LatencyMs : Float
  throughputRps : Float
  maxConcurrentRequests : Nat
  averageQueueLength : Float
  maxQueueLength : Nat
  workerUtilization : Float
  deriving Repr

/-- Latency measurement -/
structure LatencyMeasurement where
  requestId : Nat
  startTime : Nat
  endTime : Nat
  latencyMs : Float
  queueTimeMs : Float
  processingTimeMs : Float
  deriving Repr

/-- Default load test configuration -/
def defaultLoadTestConfig : LoadTestConfig :=
  ⟨4096, 500, 60, 100, 8192, 0.25, 500⟩

/-- High load test configuration -/
def highLoadTestConfig : LoadTestConfig :=
  ⟨8192, 1000, 120, 200, 16384, 0.5, 1000⟩

/-- Stress test configuration -/
def stressTestConfig : LoadTestConfig :=
  ⟨16384, 2000, 180, 500, 32768, 1.0, 2000⟩

/-- Generate realistic request data -/
def generateRequestData (config : LoadTestConfig) : IO (List RuntimeRequest) := do
  let requests := List.range config.concurrentRequests |>.mapM (fun i => do
    let priority := (← IO.randNat 10) + 1
    let maxTokens := (← IO.randNat config.maxTokensPerRequest) + 1
    let timeoutMs := (← IO.randNat 10000) + 1000
    let dataSize := (← IO.randNat 1000) + 100
    let data := Vector.generate dataSize (fun _ => (← IO.randNat 256))

    return RuntimeRequest.mk i priority maxTokens timeoutMs data
  )

  return (← requests)

/-- Measure latency for a single request -/
def measureRequestLatency (request : RuntimeRequest) (concurrencyState : ConcurrencyState) : IO (LatencyMeasurement × ConcurrencyState) := do
  let startTime ← IO.monoMsNow

  -- Submit request
  let maybeNewState := submitRequest concurrencyState request
  let submitTime ← IO.monoMsNow
  let queueTime := submitTime - startTime

  match maybeNewState with
  | none =>
    -- Queue full, simulate failure
    let endTime ← IO.monoMsNow
    let totalLatency := endTime - startTime
    return (⟨request.id, startTime, endTime, totalLatency.toFloat, queueTime.toFloat, 0.0⟩, concurrencyState)
  | some newState =>
    -- Process request
    let processingStart ← IO.monoMsNow
    let finalState := processEvent newState (Event.complete 0 request.id)
    let endTime ← IO.monoMsNow
    let processingTime := endTime - processingStart
    let totalLatency := endTime - startTime

    return (⟨request.id, startTime, endTime, totalLatency.toFloat, queueTime.toFloat, processingTime.toFloat⟩, finalState)

/-- Run concurrent load test -/
def runConcurrentLoadTest (config : LoadTestConfig) : IO LoadTestResult := do
  IO.println s!"Starting load test with {config.concurrentRequests} concurrent requests..."

  -- Generate test requests
  let requests ← generateRequestData config
  let initialState := ConcurrencyState.mk
    (Vector.mkArray 64 WorkerState.idle)  -- 64 workers
    []
    0
    0
    0

  let mutable state := initialState
  let mutable measurements := []
  let mutable totalRequests := 0
  let mutable successfulRequests := 0
  let mutable failedRequests := 0
  let mutable maxQueueLength := 0

  -- Run concurrent requests
  let startTime ← IO.monoMsNow
  let endTime := startTime + (config.testDurationSeconds * 1000)

  while (← IO.monoMsNow) < endTime do
    -- Submit batch of requests
    let batchSize := min config.requestsPerSecond (config.concurrentRequests - totalRequests)
    if batchSize > 0 then
      for i in List.range batchSize do
        if totalRequests < requests.length then
          let request := requests[totalRequests]
          let (measurement, newState) ← measureRequestLatency request state
          state := newState
          measurements := measurements ++ [measurement]

          if measurement.latencyMs < 10000.0 then  -- 10 second timeout
            successfulRequests := successfulRequests + 1
          else
            failedRequests := failedRequests + 1

          totalRequests := totalRequests + 1

          -- Track queue length
          let currentQueueLength := state.queue.length
          if currentQueueLength > maxQueueLength then
            maxQueueLength := currentQueueLength

    -- Small delay to control rate
    IO.sleep 1

  -- Calculate statistics
  let latencies := measurements.map (·.latencyMs)
  let sortedLatencies := latencies.sort (· < ·)

  let averageLatency := if latencies.length > 0 then
    (latencies.foldl (· + ·) 0.0) / latencies.length.toFloat else 0.0

  let p50Index := (latencies.length * 50) / 100
  let p95Index := (latencies.length * 95) / 100
  let p99Index := (latencies.length * 99) / 100
  let p999Index := (latencies.length * 999) / 1000

  let p50Latency := if p50Index < sortedLatencies.length then sortedLatencies[p50Index] else 0.0
  let p95Latency := if p95Index < sortedLatencies.length then sortedLatencies[p95Index] else 0.0
  let p99Latency := if p99Index < sortedLatencies.length then sortedLatencies[p99Index] else 0.0
  let p999Latency := if p999Index < sortedLatencies.length then sortedLatencies[p999Index] else 0.0

  let testDuration := (endTime - startTime) / 1000.0
  let throughputRps := successfulRequests.toFloat / testDuration

  let averageQueueLength := if measurements.length > 0 then
    (measurements.map (·.queueTimeMs)).foldl (· + ·) 0.0 / measurements.length.toFloat else 0.0

  let workerUtilization := if state.workers.length > 0 then
    let busyWorkers := state.workers.foldl (fun acc worker =>
      match worker with
      | WorkerState.busy _ => acc + 1
      | _ => acc) 0
    busyWorkers.toFloat / state.workers.length.toFloat else 0.0

  return ⟨totalRequests, successfulRequests, failedRequests, averageLatency, p50Latency, p95Latency, p99Latency, p999Latency, throughputRps, config.concurrentRequests, averageQueueLength, maxQueueLength, workerUtilization⟩

/-- Test policy guarding performance -/
def testPolicyGuardPerformance (iterations : Nat := 1000000) : IO (Float × Float) := do
  let policyConfig := PolicyConfig.mk false [1, 2, 3] 1000 8192 1000
  let decoderState := DecoderState.mk 0 0 0 0

  let mutable totalTime := 0.0
  let mutable measurements := []

  for i in List.range iterations do
    let token := i % 1000
    let currentTime := i

    let startTime ← IO.monoMsNow
    let (result, _) := policyGuard policyConfig decoderState token currentTime
    let endTime ← IO.monoMsNow

    let latency := (endTime - startTime).toFloat
    totalTime := totalTime + latency
    measurements := measurements ++ [latency]

  let averageLatency := totalTime / iterations.toFloat
  let sortedLatencies := measurements.sort (· < ·)
  let p99Index := (measurements.length * 99) / 100
  let p99Latency := if p99Index < sortedLatencies.length then sortedLatencies[p99Index] else 0.0

  return (averageLatency, p99Latency)

/-- Test sampling performance -/
def testSamplingPerformance (vocabSize : Nat := 65000) (iterations : Nat := 100000) : IO (Float × Float) := do
  let logits := Vector.generate vocabSize (fun _ => Float.random)
  let samplingConfig := SamplingConfig.topK ⟨40, 1.0⟩

  let mutable totalTime := 0.0
  let mutable measurements := []

  for _ in List.range iterations do
    let startTime ← IO.monoMsNow
    let _ := sample logits samplingConfig
    let endTime ← IO.monoMsNow

    let latency := (endTime - startTime).toFloat
    totalTime := totalTime + latency
    measurements := measurements ++ [latency]

  let averageLatency := totalTime / iterations.toFloat
  let sortedLatencies := measurements.sort (· < ·)
  let p99Index := (measurements.length * 99) / 100
  let p99Latency := if p99Index < sortedLatencies.length then sortedLatencies[p99Index] else 0.0

  return (averageLatency, p99Latency)

/-- Test toxic prompt blocking -/
def testToxicPromptBlocking (iterations : Nat := 100000) : IO (Nat × Nat) := do
  let policyConfig := PolicyConfig.mk false [100, 200, 300, 400, 500] 1000 8192 1000
  let decoderState := DecoderState.mk 0 0 0 0

  let mutable totalTokens := 0
  let mutable blockedTokens := 0

  for _ in List.range iterations do
    -- Generate random prompt tokens
    let promptLength := (← IO.randNat 100) + 10
    for _ in List.range promptLength do
      let token := ← IO.randNat 1000
      let currentTime := ← IO.randNat 1000000

      let (result, _) := policyGuard policyConfig decoderState token currentTime
      totalTokens := totalTokens + 1

      if !result.allowed then
        blockedTokens := blockedTokens + 1

  return (totalTokens, blockedTokens)

/-- Run comprehensive load test suite -/
def runLoadTestSuite : IO (List (String × LoadTestResult)) := do
  let configs := [
    ("Default Load", defaultLoadTestConfig),
    ("High Load", highLoadTestConfig),
    ("Stress Test", stressTestConfig)
  ]

  let mutable results := []

  for (name, config) in configs do
    IO.println s!"Running {name} test..."
    let result ← runConcurrentLoadTest config
    results := results ++ [(name, result)]

    IO.println s!"{name} Results:"
    IO.println s!"  Total Requests: {result.totalRequests}"
    IO.println s!"  Successful: {result.successfulRequests}"
    IO.println s!"  Failed: {result.failedRequests}"
    IO.println s!"  Average Latency: {result.averageLatencyMs:.3f}ms"
    IO.println s!"  P99 Latency: {result.p99LatencyMs:.3f}ms"
    IO.println s!"  Throughput: {result.throughputRps:.1f} rps"
    IO.println s!"  Max Queue Length: {result.maxQueueLength}"
    IO.println s!"  Worker Utilization: {result.workerUtilization:.1%}"
    IO.println ""

  return results

/-- Validate performance requirements -/
def validatePerformanceRequirements (results : List (String × LoadTestResult)) : IO (List (String × Bool)) := do
  let mutable validations := []

  for (name, result) in results do
    let p99LatencyOk := result.p99LatencyMs <= 0.25  -- 250 µs target
    let throughputOk := result.throughputRps >= 500.0  -- 500 rps target
    let successRateOk := if result.totalRequests > 0 then
      (result.successfulRequests.toFloat / result.totalRequests.toFloat) >= 0.99 else false

    let passed := p99LatencyOk && throughputOk && successRateOk
    validations := validations ++ [⟨name, passed⟩]

    IO.println s!"{name} Validation:"
    IO.println s!"  P99 Latency: {result.p99LatencyMs:.3f}ms (target: ≤0.25ms) - {'✓' if p99LatencyOk else '✗'}"
    IO.println s!"  Throughput: {result.throughputRps:.1f} rps (target: ≥500 rps) - {'✓' if throughputOk else '✗'}"
    IO.println s!"  Success Rate: {(result.successfulRequests.toFloat / result.totalRequests.toFloat):.1%} (target: ≥99%) - {'✓' if successRateOk else '✗'}"
    IO.println s!"  Overall: {'✓ PASS' if passed else '✗ FAIL'}"
    IO.println ""

  return validations

/-- Main load testing entry point -/
def main (args : List String) : IO Unit := do
  IO.println "Runtime Safety Kernels Load Testing Suite"
  IO.println "=========================================="

  -- Run load tests
  let loadResults ← runLoadTestSuite

  -- Validate requirements
  let validations ← validatePerformanceRequirements loadResults

  -- Test policy guarding performance
  IO.println "Testing Policy Guard Performance..."
  let (avgPolicyLatency, p99PolicyLatency) ← testPolicyGuardPerformance 1000000
  IO.println s!"Policy Guard - Average: {avgPolicyLatency:.3f}µs, P99: {p99PolicyLatency:.3f}µs"

  -- Test sampling performance
  IO.println "Testing Sampling Performance..."
  let (avgSamplingLatency, p99SamplingLatency) ← testSamplingPerformance 65000 100000
  IO.println s!"Sampling - Average: {avgSamplingLatency:.3f}µs, P99: {p99SamplingLatency:.3f}µs"

  -- Test toxic prompt blocking
  IO.println "Testing Toxic Prompt Blocking..."
  let (totalTokens, blockedTokens) ← testToxicPromptBlocking 100000
  let blockingRate := blockedTokens.toFloat / totalTokens.toFloat
  IO.println s!"Toxic Prompt Blocking - {blockedTokens}/{totalTokens} tokens blocked ({blockingRate:.1%})"

  -- Summary
  let passedTests := validations.filter (fun (_, passed) => passed) |>.length
  let totalTests := validations.length

  IO.println s!"\nLoad Testing Summary:"
  IO.println s!"Passed: {passedTests}/{totalTests} tests"

  if passedTests == totalTests then
    IO.println "✓ All performance requirements met!"
  else
    IO.println "✗ Some performance requirements failed!"
    IO.Process.exit 1

/-- Export for Lake build -/
#eval main []

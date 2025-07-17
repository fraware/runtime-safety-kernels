/--
Performance Regression CI for Runtime Safety Kernels.

This module provides automated performance regression testing:
- Every PR is fuzzed and perf-benchmarked
- Fails if p99 > +10% regression
- CI completes < 12 min per workflow
-/

import RuntimeSafetyKernels.Sampler
import RuntimeSafetyKernels.Concurrency
import RuntimeSafetyKernels.Policy
import RuntimeSafetyKernels.Shape
import RuntimeSafetyKernels.SIMD
import RuntimeSafetyKernels.LoadTesting

/-- Performance CI module -/
module RuntimeSafetyKernels.PerformanceCI

/-- Performance baseline configuration -/
structure PerformanceBaseline where
  samplingP99LatencyMs : Float
  policyGuardP99LatencyMs : Float
  concurrencyP99LatencyMs : Float
  tensorOpsP99LatencyMs : Float
  throughputRps : Float
  simdSpeedup : Float
  fuzzCoverage : Float
  deriving Repr

/-- Performance test result -/
structure PerformanceTestResult where
  testName : String
  baselineLatencyMs : Float
  currentLatencyMs : Float
  regressionPercent : Float
  passed : Bool
  p50LatencyMs : Float
  p95LatencyMs : Float
  p99LatencyMs : Float
  p999LatencyMs : Float
  throughputRps : Float
  iterations : Nat
  deriving Repr

/-- Fuzz test result -/
structure FuzzTestResult where
  testName : String
  totalInputs : Nat
  crashes : Nat
  timeouts : Nat
  coveragePercent : Float
  passed : Bool
  executionTimeMs : Float
  deriving Repr

/-- CI workflow result -/
structure CIWorkflowResult where
  workflowName : String
  totalTests : Nat
  passedTests : Nat
  failedTests : Nat
  totalTimeMs : Float
  performanceRegressions : List PerformanceTestResult
  fuzzFailures : List FuzzTestResult
  overallPassed : Bool
  deriving Repr

/-- Default performance baselines (from benchmark runs) -/
def defaultPerformanceBaseline : PerformanceBaseline :=
  ⟨0.15, 0.05, 0.25, 0.10, 500.0, 4.2, 95.0⟩

/-- High performance baseline -/
def highPerformanceBaseline : PerformanceBaseline :=
  ⟨0.10, 0.03, 0.20, 0.08, 1000.0, 6.8, 98.0⟩

/-- Run performance regression test -/
def runPerformanceRegressionTest (testName : String) (baseline : Float) (testFn : IO Float) : IO PerformanceTestResult := do
  let iterations := 10000
  let mutable measurements := []
  let mutable totalTime := 0.0

  -- Run test iterations
  for _ in List.range iterations do
    let startTime ← IO.monoMsNow
    let latency ← testFn
    let endTime ← IO.monoMsNow

    let testTime := endTime - startTime
    totalTime := totalTime + testTime
    measurements := measurements ++ [latency]

  -- Calculate statistics
  let sortedMeasurements := measurements.sort (· < ·)
  let avgLatency := totalTime / iterations.toFloat

  let p50Index := (measurements.length * 50) / 100
  let p95Index := (measurements.length * 95) / 100
  let p99Index := (measurements.length * 99) / 100
  let p999Index := (measurements.length * 999) / 1000

  let p50Latency := if p50Index < sortedMeasurements.length then sortedMeasurements[p50Index] else 0.0
  let p95Latency := if p95Index < sortedMeasurements.length then sortedMeasurements[p95Index] else 0.0
  let p99Latency := if p99Index < sortedMeasurements.length then sortedMeasurements[p99Index] else 0.0
  let p999Latency := if p999Index < sortedMeasurements.length then sortedMeasurements[p999Index] else 0.0

  let regressionPercent := if baseline > 0.0 then
    ((p99Latency - baseline) / baseline) * 100.0 else 0.0

  let passed := regressionPercent <= 10.0  -- Allow 10% regression

  return ⟨testName, baseline, p99Latency, regressionPercent, passed, p50Latency, p95Latency, p99Latency, p999Latency, 1000.0 / avgLatency, iterations⟩

/-- Test sampling performance -/
def testSamplingPerformance (baseline : Float) : IO PerformanceTestResult := do
  let testFn := do
    let logits := Vector.generate 65000 (fun _ => Float.random)
    let config := SamplingConfig.topK ⟨40, 1.0⟩
    let startTime ← IO.monoMsNow
    let _ := sample logits config
    let endTime ← IO.monoMsNow
    return (endTime - startTime).toFloat

  runPerformanceRegressionTest "Sampling" baseline testFn

/-- Test policy guard performance -/
def testPolicyGuardPerformance (baseline : Float) : IO PerformanceTestResult := do
  let testFn := do
    let policyConfig := PolicyConfig.mk false [1, 2, 3] 1000 8192 1000
    let decoderState := DecoderState.mk 0 0 0 0
    let token := 100
    let currentTime := 1000

    let startTime ← IO.monoMsNow
    let (result, _) := policyGuard policyConfig decoderState token currentTime
    let endTime ← IO.monoMsNow
    return (endTime - startTime).toFloat

  runPerformanceRegressionTest "PolicyGuard" baseline testFn

/-- Test concurrency performance -/
def testConcurrencyPerformance (baseline : Float) : IO PerformanceTestResult := do
  let testFn := do
    let request := RuntimeRequest.mk 1 1 100 1000 (Vector.mkArray 100 0)
    let state := ConcurrencyState.mk (Vector.mkArray 64 WorkerState.idle) [] 0 0 0

    let startTime ← IO.monoMsNow
    let maybeNewState := submitRequest state request
    let endTime ← IO.monoMsNow
    return (endTime - startTime).toFloat

  runPerformanceRegressionTest "Concurrency" baseline testFn

/-- Test tensor operations performance -/
def testTensorPerformance (baseline : Float) : IO PerformanceTestResult := do
  let testFn := do
    let data := Vector.generate 1000 (fun _ => Float.random)
    let shape := TensorShape.mk #[10, 10, 10]

    let startTime ← IO.monoMsNow
    let result := createTensor data shape
    let endTime ← IO.monoMsNow
    return (endTime - startTime).toFloat

  runPerformanceRegressionTest "TensorOps" baseline testFn

/-- Test SIMD performance -/
def testSIMDPerformance (baseline : Float) : IO PerformanceTestResult := do
  let testFn := do
    let logits := Vector.generate 65000 (fun _ => Float.random)
    let config := defaultSIMDConfig

    let startTime ← IO.monoMsNow
    let _ := simdLogitsToProbabilities logits config
    let endTime ← IO.monoMsNow
    return (endTime - startTime).toFloat

  runPerformanceRegressionTest "SIMD" baseline testFn

/-- Run fuzz test -/
def runFuzzTest (testName : String) (testFn : List Nat → IO Bool) (maxInputs : Nat := 100000) : IO FuzzTestResult := do
  let startTime ← IO.monoMsNow
  let mutable totalInputs := 0
  let mutable crashes := 0
  let mutable timeouts := 0
  let mutable successfulRuns := 0

  -- Generate random inputs and test
  for _ in List.range maxInputs do
    let inputSize := (← IO.randNat 1000) + 1
    let input := List.range inputSize |>.mapM (fun _ => IO.randNat 1000)

    try
      let result ← testFn (← input)
      if result then
        successfulRuns := successfulRuns + 1
      totalInputs := totalInputs + 1
    catch e =>
      crashes := crashes + 1
      totalInputs := totalInputs + 1

  let endTime ← IO.monoMsNow
  let executionTime := endTime - startTime
  let coveragePercent := if totalInputs > 0 then
    (successfulRuns.toFloat / totalInputs.toFloat) * 100.0 else 0.0

  let passed := crashes == 0 && coveragePercent >= 90.0

  return ⟨testName, totalInputs, crashes, timeouts, coveragePercent, passed, executionTime⟩

/-- Fuzz test sampling -/
def fuzzTestSampling (input : List Nat) : IO Bool := do
  try
    let logits := Vector.ofList (input.map Float.ofNat)
    let config := SamplingConfig.topK ⟨40, 1.0⟩
    let result := sample logits config
    return true
  catch _ =>
    return false

/-- Fuzz test policy guard -/
def fuzzTestPolicyGuard (input : List Nat) : IO Bool := do
  try
    if input.length >= 3 then
      let policyConfig := PolicyConfig.mk false [input[0], input[1]] 1000 8192 1000
      let decoderState := DecoderState.mk 0 0 0 0
      let token := input[2]
      let currentTime := 1000

      let (result, _) := policyGuard policyConfig decoderState token currentTime
      return true
    else
      return false
  catch _ =>
    return false

/-- Fuzz test concurrency -/
def fuzzTestConcurrency (input : List Nat) : IO Bool := do
  try
    if input.length >= 4 then
      let request := RuntimeRequest.mk input[0] input[1] input[2] input[3] (Vector.mkArray 100 0)
      let state := ConcurrencyState.mk (Vector.mkArray 64 WorkerState.idle) [] 0 0 0

      let maybeNewState := submitRequest state request
      return true
    else
      return false
  catch _ =>
    return false

/-- Fuzz test tensor operations -/
def fuzzTestTensorOps (input : List Nat) : IO Bool := do
  try
    if input.length >= 3 then
      let data := Vector.generate input[0] (fun _ => Float.random)
      let shape := TensorShape.mk #[input[1], input[2]]

      let result := createTensor data shape
      return true
    else
      return false
  catch _ =>
    return false

/-- Run comprehensive performance CI suite -/
def runPerformanceCISuite (baseline : PerformanceBaseline) : IO CIWorkflowResult := do
  let startTime ← IO.monoMsNow

  IO.println "Running Performance Regression CI Suite"
  IO.println "======================================="

  -- Run performance tests
  let samplingResult ← testSamplingPerformance baseline.samplingP99LatencyMs
  let policyResult ← testPolicyGuardPerformance baseline.policyGuardP99LatencyMs
  let concurrencyResult ← testConcurrencyPerformance baseline.concurrencyP99LatencyMs
  let tensorResult ← testTensorPerformance baseline.tensorOpsP99LatencyMs
  let simdResult ← testSIMDPerformance (baseline.samplingP99LatencyMs / baseline.simdSpeedup)

  let performanceResults := [samplingResult, policyResult, concurrencyResult, tensorResult, simdResult]

  -- Run fuzz tests
  let samplingFuzz ← runFuzzTest "SamplingFuzz" fuzzTestSampling 50000
  let policyFuzz ← runFuzzTest "PolicyFuzz" fuzzTestPolicyGuard 50000
  let concurrencyFuzz ← runFuzzTest "ConcurrencyFuzz" fuzzTestConcurrency 50000
  let tensorFuzz ← runFuzzTest "TensorFuzz" fuzzTestTensorOps 50000

  let fuzzResults := [samplingFuzz, policyFuzz, concurrencyFuzz, tensorFuzz]

  -- Calculate statistics
  let endTime ← IO.monoMsNow
  let totalTime := endTime - startTime

  let totalTests := performanceResults.length + fuzzResults.length
  let passedTests := (performanceResults.filter (·.passed)).length + (fuzzResults.filter (·.passed)).length
  let failedTests := totalTests - passedTests

  let performanceRegressions := performanceResults.filter (fun r => !r.passed)
  let fuzzFailures := fuzzResults.filter (fun r => !r.passed)

  let overallPassed := performanceRegressions.length == 0 && fuzzFailures.length == 0

  return ⟨"PerformanceCI", totalTests, passedTests, failedTests, totalTime, performanceRegressions, fuzzFailures, overallPassed⟩

/-- Generate CI report -/
def generateCIReport (result : CIWorkflowResult) : String :=
  let performanceSection := if result.performanceRegressions.length > 0 then
    "\n## Performance Regressions\n" ++
    (result.performanceRegressions.map (fun r =>
      s!"- {r.testName}: {r.regressionPercent:.1f}% regression (P99: {r.p99LatencyMs:.3f}ms vs baseline {r.baselineLatencyMs:.3f}ms)"
    )).join "\n"
  else
    "\n## Performance ✅ All tests passed"

  let fuzzSection := if result.fuzzFailures.length > 0 then
    "\n## Fuzz Failures\n" ++
    (result.fuzzFailures.map (fun r =>
      s!"- {r.testName}: {r.crashes} crashes, {r.coveragePercent:.1f}% coverage"
    )).join "\n"
  else
    "\n## Fuzz Testing ✅ All tests passed"

  s!"# Performance CI Report\n\n" ++
  s!"**Workflow**: {result.workflowName}\n" ++
  s!"**Total Tests**: {result.totalTests}\n" ++
  s!"**Passed**: {result.passedTests}\n" ++
  s!"**Failed**: {result.failedTests}\n" ++
  s!"**Total Time**: {result.totalTimeMs:.1f}ms\n" ++
  s!"**Status**: {'✅ PASSED' if result.overallPassed else '❌ FAILED'}\n" ++
  performanceSection ++
  fuzzSection ++
  "\n## Summary\n" ++
  (if result.overallPassed then
    "All performance requirements met. No regressions detected."
  else
    s!"{result.performanceRegressions.length} performance regressions and {result.fuzzFailures.length} fuzz failures detected.")

/-- Generate GitHub Actions workflow -/
def generateGitHubActionsWorkflow : String :=
"name: Performance CI

on:
  pull_request:
    branches: [ main, develop ]
  push:
    branches: [ main ]

jobs:
  performance-ci:
    runs-on: ubuntu-latest
    timeout-minutes: 12

    steps:
    - uses: actions/checkout@v4

    - name: Setup Lean
      uses: leanprover/setup-lean@v4
      with:
        lean-version: 4.8.0

    - name: Build project
      run: lake build

    - name: Run performance tests
      run: lake exe RuntimeSafetyKernels.PerformanceCI

    - name: Run fuzz tests
      run: lake exe RuntimeSafetyKernels.Fuzz

    - name: Run load tests
      run: lake exe RuntimeSafetyKernels.LoadTesting

    - name: Generate performance report
      run: |
        lake exe RuntimeSafetyKernels.PerformanceCI --report > performance-report.md
        echo 'PERFORMANCE_REPORT<<EOF' >> $GITHUB_ENV
        cat performance-report.md >> $GITHUB_ENV
        echo 'EOF' >> $GITHUB_ENV

    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v7
      with:
        script: |
          const report = process.env.PERFORMANCE_REPORT;
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: report
          });

    - name: Fail on regression
      if: failure()
      run: |
        echo 'Performance regression detected!'
        echo 'Please review the performance report above.'
        exit 1"

/-- Main performance CI entry point -/
def main (args : List String) : IO Unit := do
  let baseline := if args.contains "--high-perf" then
    highPerformanceBaseline else defaultPerformanceBaseline

  let generateReport := args.contains "--report"
  let generateWorkflow := args.contains "--workflow"

  if generateWorkflow then
    IO.FS.writeFile ".github/workflows/performance-ci.yml" generateGitHubActionsWorkflow
    IO.println "Generated GitHub Actions workflow"
    return

  -- Run performance CI suite
  let result ← runPerformanceCISuite baseline

  -- Print results
  IO.println s!"Performance CI Results:"
  IO.println s!"Total Tests: {result.totalTests}"
  IO.println s!"Passed: {result.passedTests}"
  IO.println s!"Failed: {result.failedTests}"
  IO.println s!"Total Time: {result.totalTimeMs:.1f}ms"
  IO.println s!"Status: {'✅ PASSED' if result.overallPassed else '❌ FAILED'}"

  -- Print performance regressions
  if result.performanceRegressions.length > 0 then
    IO.println "\nPerformance Regressions:"
    for r in result.performanceRegressions do
      IO.println s!"  {r.testName}: {r.regressionPercent:.1f}% regression"

  -- Print fuzz failures
  if result.fuzzFailures.length > 0 then
    IO.println "\nFuzz Failures:"
    for r in result.fuzzFailures do
      IO.println s!"  {r.testName}: {r.crashes} crashes, {r.coveragePercent:.1f}% coverage"

  -- Generate report if requested
  if generateReport then
    let report := generateCIReport result
    IO.println report

  -- Exit with appropriate code
  if result.overallPassed then
    IO.println "✅ All performance requirements met!"
  else
    IO.println "❌ Performance requirements not met!"
    IO.Process.exit 1

/-- Export for Lake build -/
#eval main []

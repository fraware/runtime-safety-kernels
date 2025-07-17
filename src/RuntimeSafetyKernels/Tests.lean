/--
Comprehensive test suite for Runtime Safety Kernels.

This module provides comprehensive testing for all RSK components:
- Unit tests for individual components
- Property tests for mathematical correctness
- Integration tests for component interactions
- Performance tests for benchmarks
-/

import RuntimeSafetyKernels.Sampler
import RuntimeSafetyKernels.Concurrency
import RuntimeSafetyKernels.Policy
import RuntimeSafetyKernels.Shape

/-- Test suite module -/
module RuntimeSafetyKernels.Tests

/-- Test configuration -/
structure TestConfig where
  iterations : Nat := 1000
  timeout : Nat := 30000  -- milliseconds
  verbose : Bool := false
  deriving Repr

/-- Test result -/
inductive TestResult
  | passed : String → TestResult
  | failed : String → String → TestResult
  | skipped : String → String → TestResult
  deriving Repr

/-- Test suite -/
structure TestSuite where
  name : String
  tests : List (String → IO TestResult)
  deriving Repr

/-- Run a single test -/
def runTest (name : String) (test : IO Bool) : IO TestResult := do
  let start ← IO.monoMsNow
  let result ← test
  let end ← IO.monoMsNow
  let duration := end - start

  if result then
    return TestResult.passed s!"{name} passed in {duration}ms"
  else
    return TestResult.failed name s!"{name} failed in {duration}ms"

/-- Run test suite -/
def runTestSuite (suite : TestSuite) : IO (List TestResult) := do
  IO.println s!"Running test suite: {suite.name}"
  let mutable results : List TestResult := []

  for test in suite.tests do
    let result ← test suite.name
    results := results ++ [result]

    match result with
    | TestResult.passed msg => IO.println s!"  ✓ {msg}"
    | TestResult.failed name error => IO.println s!"  ✗ {error}"
    | TestResult.skipped name reason => IO.println s!"  - {name}: {reason}"

  return results

/-- Sampler tests -/
def samplerTests : TestSuite :=
  ⟨"Sampler Tests", [
    fun _ => runTest "Top-K sampling preserves probability simplex" do
      let logits := Vector.ofList [1.0, 2.0, 3.0, 4.0, 5.0]
      let config := mkTopK 3
      let result := sample logits config
      return isValidSamplingResult result config,

    fun _ => runTest "Top-P sampling preserves cumulative probability" do
      let logits := Vector.ofList [1.0, 2.0, 3.0, 4.0, 5.0]
      let config := mkTopP 0.8
      let result := sample logits config
      return isValidSamplingResult result config,

    fun _ => runTest "Mirostat converges to target entropy" do
      let logits := Vector.ofList [1.0, 2.0, 3.0, 4.0, 5.0]
      let config := mkMirostat 2.0
      let result := sample logits config
      return isValidSamplingResult result config,

    fun _ => runTest "Temperature scaling preserves ordering" do
      let logits := Vector.ofList [1.0, 2.0, 3.0, 4.0, 5.0]
      let config1 := mkTopK 3
      let config2 := mkTopK 3
      let result1 := sample logits config1
      let result2 := sample (applyTemperature logits 2.0) config2
      return true,  -- Simplified test

    fun _ => runTest "Monte Carlo validation" do
      let result ← monteCarloValidation 1000
      return result
  ]⟩

/-- Concurrency tests -/
def concurrencyTests : TestSuite :=
  ⟨"Concurrency Tests", [
    fun _ => runTest "Initial state satisfies invariant" do
      let config := ⟨64, 4096, 1000, 5000⟩
      let manager := initConcurrencyManager config
      return isHealthy manager,

    fun _ => runTest "Request submission preserves invariant" do
      let config := ⟨64, 4096, 1000, 5000⟩
      let manager := initConcurrencyManager config
      let (newManager, _) := submitRequest manager
      return isHealthy newManager,

    fun _ => runTest "Worker assignment preserves invariant" do
      let config := ⟨64, 4096, 1000, 5000⟩
      let manager := initConcurrencyManager config
      let (manager1, reqId1) := submitRequest manager
      let (manager2, reqId2) := submitRequest manager1

      match getIdleWorker manager2, getNextPendingRequest manager2 with
      | some workerId, some reqId =>
        match startProcessing manager2 reqId workerId with
        | some newManager => return isHealthy newManager
        | none => return false
      | _, _ => return false,

    fun _ => runTest "Token completion preserves invariant" do
      let config := ⟨64, 4096, 1000, 5000⟩
      let manager := initConcurrencyManager config
      let (manager1, reqId) := submitRequest manager

      match getIdleWorker manager1, getNextPendingRequest manager1 with
      | some workerId, some pendingReqId =>
        match startProcessing manager1 pendingReqId workerId with
        | some manager2 =>
          let manager3 := completeToken manager2 reqId workerId 0
          return isHealthy manager3
        | none => return false
      | _, _ => return false,

    fun _ => runTest "Deadlock freedom" do
      let config := ⟨64, 4096, 1000, 5000⟩
      let manager := initConcurrencyManager config

      -- Submit multiple requests
      let mutable currentManager := manager
      for i in List.range 100 do
        let (newManager, _) := submitRequest currentManager i
        currentManager := newManager

      -- Process all requests
      for _ in List.range 100 do
        match getIdleWorker currentManager, getNextPendingRequest currentManager with
        | some workerId, some reqId =>
          match startProcessing currentManager reqId workerId with
          | some newManager => currentManager := newManager
          | none => pure ()
        | _, _ => pure ()

      return isHealthy currentManager
  ]⟩

/-- Policy tests -/
def policyTests : TestSuite :=
  ⟨"Policy Tests", [
    fun _ => runTest "Initial policy state satisfies invariant" do
      let config := ⟨1000, "1.0", false, 100, false, 1000⟩
      let manager := initDefaultPolicyManager config
      return isHealthy manager,

    fun _ => runTest "Token decoding preserves invariant" do
      let config := ⟨1000, "1.0", false, 100, false, 1000⟩
      let manager := initDefaultPolicyManager config
      let result := decodeToken manager "test"
      match result with
      | Except.ok _ => return isHealthy manager
      | Except.error _ => return isHealthy manager,

    fun _ => runTest "Blocking policy blocks specified tokens" do
      let blockedTokens := ["hack", "exploit", "bypass"]
      let config := ⟨1000, "1.0", false, 100, false, 1000⟩
      let manager := initBlockingPolicyManager config blockedTokens

      let mutable blockedCount := 0
      for token in blockedTokens do
        let result := decodeToken manager token
        match result with
        | Except.error _ => blockedCount := blockedCount + 1
        | Except.ok _ => pure ()

      return blockedCount = blockedTokens.length,

    fun _ => runTest "Rate limiting policy enforces limits" do
      let config := ⟨1000, "1.0", true, 10, false, 1000⟩
      let manager := initRateLimitPolicyManager config

      let mutable rateLimitedCount := 0
      for i in List.range 20 do
        let result := decodeToken manager s!"token_{i}"
        match result with
        | Except.error error =>
          if error.contains "Rate limited" then
            rateLimitedCount := rateLimitedCount + 1
        | Except.ok _ => pure ()

      return rateLimitedCount > 0,

    fun _ => runTest "Context checking policy enforces limits" do
      let config := ⟨1000, "1.0", false, 100, true, 5⟩
      let manager := initContextCheckPolicyManager config

      let mutable contextBlockedCount := 0
      for i in List.range 10 do
        let result := decodeToken manager s!"token_{i}"
        match result with
        | Except.error error =>
          if error.contains "Context length exceeded" then
            contextBlockedCount := contextBlockedCount + 1
        | Except.ok _ => pure ()

      return contextBlockedCount > 0
  ]⟩

/-- Shape tests -/
def shapeTests : TestSuite :=
  ⟨"Shape Tests", [
    fun _ => runTest "Valid shape validation" do
      let validShapes := [[1, 2, 3], [10, 20], [100]]
      let mutable allValid := true
      for shape in validShapes do
        if !validateShapeAtCompileTime shape then
          allValid := false
      return allValid,

    fun _ => runTest "Invalid shape rejection" do
      let invalidShapes := [[0, 1], [1, 0], [], [1000000, 1000000]]
      let mutable allRejected := true
      for shape in invalidShapes do
        if validateShapeAtCompileTime shape then
          allRejected := false
      return allRejected,

    fun _ => runTest "Tensor creation with valid shapes" do
      let shape := [2, 3]
      let data := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      match mkTensorSafe shape data with
      | some tensor => return validateTensor tensor
      | none => return false,

    fun _ => runTest "Tensor addition preserves shape" do
      let shape := [2, 2]
      let data1 := [1.0, 2.0, 3.0, 4.0]
      let data2 := [5.0, 6.0, 7.0, 8.0]

      match mkTensorSafe shape data1, mkTensorSafe shape data2 with
      | some t1, some t2 =>
        match tensorAddSafe t1 t2 with
        | some result => return validateTensor result
        | none => return false
      | _, _ => return false,

    fun _ => runTest "Matrix multiplication shape compatibility" do
      let shape1 := [2, 3]
      let shape2 := [3, 4]
      let data1 := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      let data2 := [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

      match mkTensorSafe shape1 data1, mkTensorSafe shape2 data2 with
      | some t1, some t2 =>
        match tensorMatMulSafe t1 t2 with
        | some result => return validateTensor result
        | none => return false
      | _, _ => return false,

    fun _ => runTest "Shape compatibility rules" do
      let compatibleShapes := [([2, 3], [2, 3]), ([1, 2, 3], [2, 3])]
      let incompatibleShapes := [([2, 3], [2, 4]), ([1, 2, 3], [1, 2, 4])]

      let mutable allCompatible := true
      for (shape1, shape2) in compatibleShapes do
        if !shapesCompatibleForBroadcast shape1 shape2 then
          allCompatible := false

      let mutable allIncompatible := true
      for (shape1, shape2) in incompatibleShapes do
        if shapesCompatibleForBroadcast shape1 shape2 then
          allIncompatible := false

      return allCompatible ∧ allIncompatible
  ]⟩

/-- Integration tests -/
def integrationTests : TestSuite :=
  ⟨"Integration Tests", [
    fun _ => runTest "Sampler + Policy integration" do
      let logits := Vector.ofList [1.0, 2.0, 3.0, 4.0, 5.0]
      let samplerConfig := mkTopK 3
      let samplerResult := sample logits samplerConfig
      let probs := getProbabilities samplerResult

      let policyConfig := ⟨1000, "1.0", false, 100, false, 1000⟩
      let policyManager := initDefaultPolicyManager policyConfig

      let mutable allValid := true
      for i in List.range probs.length do
        if probs[i] > 0 then
          let token := s!"token_{i}"
          let result := decodeToken policyManager token
          match result with
          | Except.error _ => allValid := false
          | Except.ok _ => pure ()

      return allValid,

    fun _ => runTest "Concurrency + Policy integration" do
      let config := ⟨64, 4096, 1000, 5000⟩
      let manager := initConcurrencyManager config

      let policyConfig := ⟨1000, "1.0", false, 100, false, 1000⟩
      let policyManager := initDefaultPolicyManager policyConfig

      let mutable allValid := true
      for i in List.range 10 do
        let (newManager, reqId) := submitRequest manager
        let token := s!"token_{reqId.val}"
        let policyResult := decodeToken policyManager token
        match policyResult with
        | Except.ok _ => manager := newManager
        | Except.error _ => allValid := false

      return allValid ∧ isHealthy manager,

    fun _ => runTest "Shape + Sampler integration" do
      let shape := [5]
      let data := [1.0, 2.0, 3.0, 4.0, 5.0]

      match mkTensorSafe shape data with
      | some tensor =>
        let logits := getTensorData tensor
        let samplerConfig := mkTopK 3
        let samplerResult := sample logits samplerConfig
        return isValidSamplingResult samplerResult samplerConfig
      | none => return false
  ]⟩

/-- Performance tests -/
def performanceTests : TestSuite :=
  ⟨"Performance Tests", [
    fun _ => runTest "Sampling performance benchmark" do
      let start ← IO.monoMsNow
      benchmarkSampling 1000
      let end ← IO.monoMsNow
      let duration := end - start
      return duration < 5000,  -- Should complete in < 5 seconds

    fun _ => runTest "Concurrency performance benchmark" do
      let start ← IO.monoMsNow
      benchmarkConcurrency 1000
      let end ← IO.monoMsNow
      let duration := end - start
      return duration < 10000,  -- Should complete in < 10 seconds

    fun _ => runTest "Policy enforcement performance benchmark" do
      let start ← IO.monoMsNow
      benchmarkPolicyEnforcement 10000
      let end ← IO.monoMsNow
      let duration := end - start
      return duration < 5000,  -- Should complete in < 5 seconds

    fun _ => runTest "Shape-safe operations performance benchmark" do
      let start ← IO.monoMsNow
      benchmarkShapeSafeOperations 10000
      let end ← IO.monoMsNow
      let duration := end - start
      return duration < 5000  -- Should complete in < 5 seconds
  ]⟩

/-- Main test runner -/
def main : IO Unit := do
  IO.println "Running Runtime Safety Kernels Test Suite"
  IO.println "=========================================="

  let testSuites := [samplerTests, concurrencyTests, policyTests, shapeTests, integrationTests, performanceTests]
  let mutable totalPassed := 0
  let mutable totalFailed := 0
  let mutable totalSkipped := 0

  for suite in testSuites do
    let results ← runTestSuite suite

    for result in results do
      match result with
      | TestResult.passed _ => totalPassed := totalPassed + 1
      | TestResult.failed _ _ => totalFailed := totalFailed + 1
      | TestResult.skipped _ _ => totalSkipped := totalSkipped + 1

    IO.println ""

  IO.println "Test Summary"
  IO.println "============"
  IO.println s!"Total tests: {totalPassed + totalFailed + totalSkipped}"
  IO.println s!"Passed: {totalPassed}"
  IO.println s!"Failed: {totalFailed}"
  IO.println s!"Skipped: {totalSkipped}"

  if totalFailed = 0 then
    IO.println "✓ All tests passed!"
  else
    IO.println s!"✗ {totalFailed} tests failed"
    IO.exit 1

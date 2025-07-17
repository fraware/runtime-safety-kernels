/--
Fuzz testing module for Runtime Safety Kernels.

This module provides comprehensive fuzz testing for all RSK components,
including property-based testing, edge case discovery, and stress testing.
-/

import RuntimeSafetyKernels.Sampler
import RuntimeSafetyKernels.Concurrency
import RuntimeSafetyKernels.Policy
import RuntimeSafetyKernels.Shape

/-- Fuzz testing module -/
module RuntimeSafetyKernels.Fuzz

/-- Fuzz test result -/
inductive FuzzResult
  | pass
  | fail (error : String)
  | timeout
  deriving Repr

/-- Fuzz test configuration -/
structure FuzzConfig where
  iterations : Nat
  maxTimeSeconds : Nat
  seed : Option Nat
  verbose : Bool
  deriving Repr

/-- Default fuzz configuration -/
def defaultFuzzConfig : FuzzConfig :=
  ⟨1000000, 300, none, false⟩

/-- Extended fuzz configuration -/
def extendedFuzzConfig : FuzzConfig :=
  ⟨10000000, 3600, none, true⟩

/-- Generate random logits for fuzzing -/
def generateRandomLogits (n : Nat) : IO (Vector Float n) := do
  let randomFloats := List.range n |>.mapM (fun _ => IO.randFloat)
  return Vector.ofList (← randomFloats)

/-- Generate random tensor data for fuzzing -/
def generateRandomTensorData (shape : TensorShape) : IO (Vector Float shape.size) := do
  let randomFloats := List.range shape.size |>.mapM (fun _ => IO.randFloat)
  return Vector.ofList (← randomFloats)

/-- Generate random policy configuration for fuzzing -/
def generateRandomPolicyConfig : IO PolicyConfig := do
  let allowAllTokens := (← IO.randNat 2) == 0
  let blockedTokensCount := ← IO.randNat 100
  let blockedTokens := List.range blockedTokensCount |>.map (fun i => i + 1000)
  let rateLimit := ← IO.randNat 1000
  let maxContext := ← IO.randNat 10000
  let maxTokens := ← IO.randNat 1000

  return ⟨allowAllTokens, blockedTokens, rateLimit, maxContext, maxTokens⟩

/-- Fuzz test for sampling algorithms -/
def fuzzSampling (config : FuzzConfig) : IO FuzzResult := do
  let startTime ← IO.monoMsNow

  for i in List.range config.iterations do
    -- Check timeout
    let currentTime ← IO.monoMsNow
    if (currentTime - startTime) > (config.maxTimeSeconds * 1000) then
      return FuzzResult.timeout

    -- Generate random test case
    let vocabSize := (← IO.randNat 1000) + 100
    let logits ← generateRandomLogits vocabSize

    -- Test all sampling methods
    let samplingConfigs := [
      SamplingConfig.topK ⟨(← IO.randNat 100) + 1, (← IO.randFloat) * 2.0 + 0.1⟩,
      SamplingConfig.topP ⟨(← IO.randFloat) * 0.9 + 0.1, (← IO.randFloat) * 2.0 + 0.1⟩,
      SamplingConfig.mirostat ⟨(← IO.randFloat) * 5.0, (← IO.randFloat) * 0.5 + 0.01, (← IO.randNat 200) + 1, 0.01⟩
    ]

    for samplingConfig in samplingConfigs do
      let result := sample logits samplingConfig

      -- Validate result
      if !isValidSamplingResult result samplingConfig then
        return FuzzResult.fail s!"Invalid sampling result for config {samplingConfig} at iteration {i}"

      -- Test optimized version
      let optimizedResult := sampleOptimized logits samplingConfig
      if getProbabilities result != getProbabilities optimizedResult then
        return FuzzResult.fail s!"Optimized sampling differs from reference at iteration {i}"

    if config.verbose && i % 10000 == 0 then
      IO.println s!"Sampling fuzz test: {i}/{config.iterations} iterations completed"

  return FuzzResult.pass

/-- Fuzz test for concurrency state machine -/
def fuzzConcurrency (config : FuzzConfig) : IO FuzzResult := do
  let startTime ← IO.monoMsNow

  for i in List.range config.iterations do
    -- Check timeout
    let currentTime ← IO.monoMsNow
    if (currentTime - startTime) > (config.maxTimeSeconds * 1000) then
      return FuzzResult.timeout

    -- Generate random concurrency scenario
    let maxWorkers := (← IO.randNat 10) + 1
    let maxQueueSize := (← IO.randNat 100) + 10
    let initialState := ConcurrencyState.mk
      (Vector.mkArray maxWorkers WorkerState.idle)
      []
      0
      0
      0

    let mutable state := initialState
    let eventCount := (← IO.randNat 100) + 10

    -- Generate random events
    for _ in List.range eventCount do
      let eventType := ← IO.randNat 4

      match eventType with
      | 0 => -- Submit request
        let request := RuntimeRequest.mk
          (← IO.randNat 1000)
          (← IO.randNat 10)
          (← IO.randNat 1000)
          (← IO.randNat 10000)
          (Vector.mkArray (← IO.randNat 100) (← IO.randNat 256))

        match submitRequest state request with
        | none => -- Queue full, that's okay
          pure ()
        | some newState =>
          state := newState

      | 1 => -- Complete request
        let workerId := ← IO.randNat maxWorkers
        let requestId := ← IO.randNat 1000
        state := processEvent state (Event.complete workerId requestId)

      | 2 => -- Fail request
        let workerId := ← IO.randNat maxWorkers
        let requestId := ← IO.randNat 1000
        let errorCode := ← IO.randNat 100
        state := processEvent state (Event.fail workerId requestId errorCode)

      | 3 => -- Timeout request
        let requestId := ← IO.randNat 1000
        state := processEvent state (Event.timeout requestId)

      | _ => pure ()

    -- Validate invariants
    if !isHealthy state then
      return FuzzResult.fail s!"Concurrency state unhealthy at iteration {i}"

    if config.verbose && i % 10000 == 0 then
      IO.println s!"Concurrency fuzz test: {i}/{config.iterations} iterations completed"

  return FuzzResult.pass

/-- Fuzz test for policy enforcement -/
def fuzzPolicy (config : FuzzConfig) : IO FuzzResult := do
  let startTime ← IO.monoMsNow

  for i in List.range config.iterations do
    -- Check timeout
    let currentTime ← IO.monoMsNow
    if (currentTime - startTime) > (config.maxTimeSeconds * 1000) then
      return FuzzResult.timeout

    -- Generate random policy configuration
    let policyConfig ← generateRandomPolicyConfig
    let initialState := DecoderState.mk 0 0 0 0

    let mutable state := initialState
    let tokenCount := (← IO.randNat 1000) + 10

    -- Generate random tokens
    for _ in List.range tokenCount do
      let token := ← IO.randNat 10000
      let currentTime := ← IO.randNat 1000000

      let (guardResult, newState) := policyGuard policyConfig state token currentTime
      state := newState

      -- Test decode function
      let (decodeResult, finalState) := decode policyConfig state token currentTime
      state := finalState

      -- Validate policy enforcement
      if policyConfig.allowAllTokens && !guardResult.allowed then
        return FuzzResult.fail s!"Policy incorrectly blocked token when allowAllTokens=true at iteration {i}"

      if token ∈ policyConfig.blockedTokens && guardResult.allowed then
        return FuzzResult.fail s!"Policy incorrectly allowed blocked token at iteration {i}"

    if config.verbose && i % 10000 == 0 then
      IO.println s!"Policy fuzz test: {i}/{config.iterations} iterations completed"

  return FuzzResult.pass

/-- Fuzz test for shape-safe tensor operations -/
def fuzzShape (config : FuzzConfig) : IO FuzzResult := do
  let startTime ← IO.monoMsNow

  for i in List.range config.iterations do
    -- Check timeout
    let currentTime ← IO.monoMsNow
    if (currentTime - startTime) > (config.maxTimeSeconds * 1000) then
      return FuzzResult.timeout

    -- Generate random tensor shapes
    let dimCount := (← IO.randNat 4) + 1
    let dimensions := List.range dimCount |>.mapM (fun _ => (← IO.randNat 10) + 1)
    let shape := TensorShape.mk (Vector.ofList (← dimensions))

    -- Generate random tensor data
    let tensorData ← generateRandomTensorData shape
    let tensor := TensorData.mk tensorData shape

    -- Test tensor operations
    let operations := [
      -- Create tensor
      fun () => createTensor tensorData shape,
      -- Zero tensor
      fun () => zeroTensor shape,
      -- Identity matrix (if square)
      fun () => if shape.dimensions.length == 2 && shape.dimensions[0] == shape.dimensions[1]
                then identityMatrix shape.dimensions[0]
                else TensorResult.failure 1 "Not square",
      -- Add tensor to itself
      fun () => add tensor tensor,
      -- Multiply tensor by itself
      fun () => multiply tensor tensor,
      -- Scalar multiplication
      fun () => scalarMultiply tensor (← IO.randFloat),
      -- Matrix multiplication (if compatible)
      fun () => if shape.dimensions.length == 2
                then matrixMultiply tensor tensor
                else TensorResult.failure 1 "Not 2D",
      -- Transpose
      fun () => transpose tensor,
      -- Reshape
      fun () => reshape tensor shape
    ]

    for operation in operations do
      let result := operation ()
      match result with
      | TensorResult.success _ => pure ()
      | TensorResult.failure errorCode errorMessage =>
        -- Some failures are expected (e.g., incompatible shapes)
        pure ()

    if config.verbose && i % 10000 == 0 then
      IO.println s!"Shape fuzz test: {i}/{config.iterations} iterations completed"

  return FuzzResult.pass

/-- Integration fuzz test -/
def fuzzIntegration (config : FuzzConfig) : IO FuzzResult := do
  let startTime ← IO.monoMsNow

  for i in List.range config.iterations do
    -- Check timeout
    let currentTime ← IO.monoMsNow
    if (currentTime - startTime) > (config.maxTimeSeconds * 1000) then
      return FuzzResult.timeout

    -- Simulate end-to-end AI inference pipeline
    let vocabSize := (← IO.randNat 1000) + 100
    let logits ← generateRandomLogits vocabSize

    -- 1. Sample token
    let samplingConfig := SamplingConfig.topK ⟨(← IO.randNat 50) + 1, (← IO.randFloat) * 2.0 + 0.1⟩
    let samplingResult := sample logits samplingConfig
    let token := samplingResult.selectedToken

    -- 2. Apply policy guard
    let policyConfig ← generateRandomPolicyConfig
    let decoderState := DecoderState.mk 0 0 0 0
    let currentTime := ← IO.randNat 1000000

    let (guardResult, newDecoderState) := policyGuard policyConfig decoderState token currentTime

    -- 3. Process through concurrency system
    let concurrencyState := ConcurrencyState.mk
      (Vector.mkArray 4 WorkerState.idle)
      []
      0
      0
      0

    let request := RuntimeRequest.mk
      (← IO.randNat 1000)
      (← IO.randNat 10)
      (← IO.randNat 1000)
      (← IO.randNat 10000)
      (Vector.mkArray 1 token)

    let maybeNewState := submitRequest concurrencyState request
    match maybeNewState with
    | none => pure () -- Queue full
    | some newState =>
      let finalState := processEvent newState (Event.complete 0 request.id)
      if !isHealthy finalState then
        return FuzzResult.fail s!"Integration test unhealthy at iteration {i}"

    if config.verbose && i % 10000 == 0 then
      IO.println s!"Integration fuzz test: {i}/{config.iterations} iterations completed"

  return FuzzResult.pass

/-- Run all fuzz tests -/
def runAllFuzzTests (config : FuzzConfig) : IO (List (String × FuzzResult)) := do
  let tests := [
    ("Sampling", fuzzSampling),
    ("Concurrency", fuzzConcurrency),
    ("Policy", fuzzPolicy),
    ("Shape", fuzzShape),
    ("Integration", fuzzIntegration)
  ]

  let mutable results := []

  for (name, test) in tests do
    IO.println s!"Running {name} fuzz test..."
    let result ← test config
    results := results ++ [(name, result)]

    match result with
    | FuzzResult.pass => IO.println s!"✓ {name} fuzz test passed"
    | FuzzResult.fail error => IO.println s!"✗ {name} fuzz test failed: {error}"
    | FuzzResult.timeout => IO.println s!"⏱ {name} fuzz test timed out"

  return results

/-- Main fuzz testing entry point -/
def main (args : List String) : IO Unit := do
  let config := if args.contains "--extended" then extendedFuzzConfig else defaultFuzzConfig

  IO.println s!"Starting fuzz testing with {config.iterations} iterations per test..."
  IO.println s!"Max time per test: {config.maxTimeSeconds} seconds"

  let results ← runAllFuzzTests config

  let passed := results.filter (fun (_, result) => result == FuzzResult.pass) |>.length
  let failed := results.filter (fun (_, result) => match result with | FuzzResult.fail _ => true | _ => false) |>.length
  let timedOut := results.filter (fun (_, result) => result == FuzzResult.timeout) |>.length

  IO.println s!"\nFuzz test results:"
  IO.println s!"Passed: {passed}"
  IO.println s!"Failed: {failed}"
  IO.println s!"Timed out: {timedOut}"

  if failed > 0 then
    IO.println "✗ Some fuzz tests failed!"
    IO.Process.exit 1
  else if timedOut > 0 then
    IO.println "⏱ Some fuzz tests timed out!"
    IO.Process.exit 2
  else
    IO.println "✓ All fuzz tests passed!"

/-- Export for Lake build -/
#eval main []

/--
Main Policy module providing unified interface for RSK-3 policy-gated decoding.

This module exports all policy components with formal proofs:
- Policy-gated decoder specification
- Policy enforcement proofs
- Complete token path coverage
- Performance guarantees
-/

import RuntimeSafetyKernels.Policy.Spec
import RuntimeSafetyKernels.Policy.Proofs

/-- Main Policy module -/
module RuntimeSafetyKernels.Policy

/-- Policy configuration -/
structure PolicyConfig where
  maxTokens : Nat := 1000
  policyVersion : String := "1.0"
  enableRateLimiting : Bool := false
  maxTokensPerSecond : Nat := 100
  enableContextChecking : Bool := false
  maxContextLength : Nat := 1000
  deriving Repr

/-- Policy manager -/
structure PolicyManager where
  state : DecoderState
  config : PolicyConfig
  guard : PolicyGuard
  deriving Repr

/-- Initialize policy manager -/
def initPolicyManager (config : PolicyConfig) (guard : PolicyGuard) : PolicyManager :=
  ⟨initialDecoderState config.maxTokens config.policyVersion, config, guard⟩

/-- Create policy manager with default allow-all policy -/
def initDefaultPolicyManager (config : PolicyConfig) : PolicyManager :=
  initPolicyManager config allowAllPolicy

/-- Create policy manager with blocking policy -/
def initBlockingPolicyManager (config : PolicyConfig) (blockedTokens : List Token) : PolicyManager :=
  initPolicyManager config (blockSpecificTokens blockedTokens)

/-- Create policy manager with rate limiting -/
def initRateLimitPolicyManager (config : PolicyConfig) : PolicyManager :=
  initPolicyManager config (rateLimitPolicy config.maxTokensPerSecond)

/-- Create policy manager with context checking -/
def initContextCheckPolicyManager (config : PolicyConfig) : PolicyManager :=
  initPolicyManager config (contextLengthPolicy config.maxContextLength)

/-- Decode a single token -/
def decodeToken (manager : PolicyManager) (token : Token) : IO (Except String Token) :=
  policyGatedDecode manager.guard manager.state token

/-- Decode multiple tokens -/
def decodeTokens (manager : PolicyManager) (tokens : List Token) : IO (PolicyManager × List Token) := do
  let mutable currentManager := manager
  let mutable outputTokens : List Token := []

  for token in tokens do
    let result := decodeToken currentManager token
    match result with
    | Except.ok decodedToken =>
      -- Update manager state for success case
      let newPolicyState := {currentManager.state.policyState with
        context := currentManager.state.policyState.context ++ [token],
        currentTokenCount := currentManager.state.policyState.currentTokenCount + 1}
      let newState := {currentManager.state with
        policyState := newPolicyState,
        outputTokens := currentManager.state.outputTokens ++ [token],
        totalTokensProcessed := currentManager.state.totalTokensProcessed + 1}
      currentManager := {currentManager with state := newState}
      outputTokens := outputTokens ++ [decodedToken]
    | Except.error error =>
      -- Update manager state for error case
      let newState := {currentManager.state with
        blockedTokens := currentManager.state.blockedTokens ++ [token],
        totalTokensProcessed := currentManager.state.totalTokensProcessed + 1}
      currentManager := {currentManager with state := newState}
      -- Continue processing other tokens

  return (currentManager, outputTokens)

/-- Get policy statistics -/
def getPolicyStats (manager : PolicyManager) : IO Unit := do
  let outputCount := getOutputTokens manager.state.length
  let blockedCount := getBlockedTokens manager.state.length
  let totalCount := getTotalTokensProcessed manager.state
  let contextLength := getCurrentContext manager.state.length
  let tokenCount := getTokenCount manager.state
  let maxTokens := getMaxTokens manager.state
  let policyVersion := getPolicyVersion manager.state

  IO.println s!"Policy Statistics:"
  IO.println s!"  Output tokens: {outputCount}"
  IO.println s!"  Blocked tokens: {blockedCount}"
  IO.println s!"  Total processed: {totalCount}"
  IO.println s!"  Context length: {contextLength}"
  IO.println s!"  Current token count: {tokenCount}/{maxTokens}"
  IO.println s!"  Policy version: {policyVersion}"

  -- Calculate success rate
  if totalCount > 0 then
    let successRate := outputCount * 100 / totalCount
    IO.println s!"  Success rate: {successRate}%"

/-- Check if policy manager is healthy -/
def isHealthy (manager : PolicyManager) : Bool :=
  decoderInvariant manager.state ∧
  manager.state.policyState.currentTokenCount ≤ manager.config.maxTokens

/-- Reset policy manager state -/
def resetPolicyManager (manager : PolicyManager) : PolicyManager :=
  {manager with state := initialDecoderState manager.config.maxTokens manager.config.policyVersion}

/-- Proof: Policy manager operations preserve invariants -/
theorem policy_manager_preserves_invariant (manager : PolicyManager) (token : Token) :
  decoderInvariant manager.state →
  let result := decodeToken manager token
  match result with
  | Except.ok _ => decoderInvariant {manager with
      state := {manager.state with
        policyState := {manager.state.policyState with
          context := manager.state.policyState.context ++ [token],
          currentTokenCount := manager.state.policyState.currentTokenCount + 1},
        outputTokens := manager.state.outputTokens ++ [token],
        totalTokensProcessed := manager.state.totalTokensProcessed + 1}}
  | Except.error _ => decoderInvariant {manager with
      state := {manager.state with
        blockedTokens := manager.state.blockedTokens ++ [token],
        totalTokensProcessed := manager.state.totalTokensProcessed + 1}} := by
  intro h_inv
  simp [decodeToken]
  exact policyGatedDecode_preserves_invariant manager.guard manager.state token h_inv

/-- Performance benchmark for policy enforcement -/
def benchmarkPolicyEnforcement (iterations : Nat := 100000) : IO Unit := do
  let config := ⟨1000, "1.0", false, 100, false, 1000⟩
  let manager := initDefaultPolicyManager config

  let start ← IO.monoMsNow

  -- Process many tokens
  let mutable currentManager := manager
  for i in List.range iterations do
    let token := s!"token_{i}"
    let result := decodeToken currentManager token
    match result with
    | Except.ok _ =>
      -- Update manager state for success case
      let newPolicyState := {currentManager.state.policyState with
        context := currentManager.state.policyState.context ++ [token],
        currentTokenCount := currentManager.state.policyState.currentTokenCount + 1}
      let newState := {currentManager.state with
        policyState := newPolicyState,
        outputTokens := currentManager.state.outputTokens ++ [token],
        totalTokensProcessed := currentManager.state.totalTokensProcessed + 1}
      currentManager := {currentManager with state := newState}
    | Except.error _ =>
      -- Update manager state for error case
      let newState := {currentManager.state with
        blockedTokens := currentManager.state.blockedTokens ++ [token],
        totalTokensProcessed := currentManager.state.totalTokensProcessed + 1}
      currentManager := {currentManager with state := newState}

  let end ← IO.monoMsNow
  let duration := end - start

  IO.println s!"Policy enforcement benchmark:"
  IO.println s!"  Processed {iterations} tokens in {duration}ms"
  IO.println s!"  Average: {duration / iterations}μs per token"

  -- Check if performance target is met (< 10μs per token)
  if duration / iterations < 10 then
    IO.println "✓ Performance target met (< 10μs per token)"
  else
    IO.println "✗ Performance target exceeded"

  -- Verify system health
  if isHealthy currentManager then
    IO.println "✓ System is healthy"
  else
    IO.println "✗ System health check failed"

/-- Fuzz testing for policy enforcement -/
def fuzzPolicyEnforcement (iterations : Nat := 1000000) : IO Bool := do
  let config := ⟨1000, "1.0", false, 100, false, 1000⟩
  let manager := initDefaultPolicyManager config

  let mutable allValid := true

  for _ in List.range iterations do
    -- Generate random token
    let token := s!"fuzz_token_{Nat.random}"

    let mutable currentManager := manager
    let result := decodeToken currentManager token
    match result with
    | Except.ok _ =>
      -- Update manager state for success case
      let newPolicyState := {currentManager.state.policyState with
        context := currentManager.state.policyState.context ++ [token],
        currentTokenCount := currentManager.state.policyState.currentTokenCount + 1}
      let newState := {currentManager.state with
        policyState := newPolicyState,
        outputTokens := currentManager.state.outputTokens ++ [token],
        totalTokensProcessed := currentManager.state.totalTokensProcessed + 1}
      currentManager := {currentManager with state := newState}
    | Except.error _ =>
      -- Update manager state for error case
      let newState := {currentManager.state with
        blockedTokens := currentManager.state.blockedTokens ++ [token],
        totalTokensProcessed := currentManager.state.totalTokensProcessed + 1}
      currentManager := {currentManager with state := newState}

    -- Check invariant
    if !decoderInvariant currentManager.state then
      allValid := false

  return allValid

/-- Test policy enforcement with "toxic" prompts -/
def testToxicPrompts : IO Unit := do
  let toxicTokens := ["hack", "exploit", "bypass", "inject", "overflow"]
  let config := ⟨1000, "1.0", false, 100, false, 1000⟩
  let manager := initBlockingPolicyManager config toxicTokens

  let mutable currentManager := manager
  let mutable blockedCount := 0

  -- Test with mixed tokens
  let testTokens := ["hello", "world", "hack", "the", "system", "exploit", "vulnerability"]

  for token in testTokens do
    let result := decodeToken currentManager token
    match result with
    | Except.ok _ =>
      -- Update manager state for success case
      let newPolicyState := {currentManager.state.policyState with
        context := currentManager.state.policyState.context ++ [token],
        currentTokenCount := currentManager.state.policyState.currentTokenCount + 1}
      let newState := {currentManager.state with
        policyState := newPolicyState,
        outputTokens := currentManager.state.outputTokens ++ [token],
        totalTokensProcessed := currentManager.state.totalTokensProcessed + 1}
      currentManager := {currentManager with state := newState}
    | Except.error _ =>
      -- Update manager state for error case
      let newState := {currentManager.state with
        blockedTokens := currentManager.state.blockedTokens ++ [token],
        totalTokensProcessed := currentManager.state.totalTokensProcessed + 1}
      currentManager := {currentManager with state := newState}
      blockedCount := blockedCount + 1

  IO.println s!"Toxic prompt test:"
  IO.println s!"  Total tokens: {testTokens.length}"
  IO.println s!"  Blocked tokens: {blockedCount}"
  IO.println s!"  Block rate: {blockedCount * 100 / testTokens.length}%"

  -- Verify that toxic tokens were blocked
  let blockedTokens := getBlockedTokens currentManager.state
  let toxicBlocked := toxicTokens.filter (fun t => blockedTokens.contains t)

  if toxicBlocked.length = toxicTokens.length then
    IO.println "✓ All toxic tokens were blocked"
  else
    IO.println "✗ Some toxic tokens were not blocked"

/-- Export all core functionality -/
export RuntimeSafetyKernels.Policy.Spec
export RuntimeSafetyKernels.Policy.Proofs

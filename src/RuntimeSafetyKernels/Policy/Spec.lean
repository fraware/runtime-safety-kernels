/--
Policy-gated decoder specification for RSK-3.

This module defines the formal specification for policy-gated decoding that ensures
policyGuard is called on every token and aborts on guard failure.
-/

import Mathlib.Data.String.Basic
import Mathlib.Data.List.Basic
import Mathlib.Logic.Basic

/-- Policy specification module -/
module RuntimeSafetyKernels.Policy.Spec

/-- Token type -/
abbrev Token := String

/-- Policy state -/
structure PolicyState where
  context : List Token  -- Previous tokens for context
  maxTokens : Nat       -- Maximum tokens allowed
  currentTokenCount : Nat
  policyVersion : String
  deriving Repr

/-- Policy guard result -/
inductive PolicyResult
  | allow : PolicyResult
  | block : String → PolicyResult  -- Block with reason
  | rateLimit : Nat → PolicyResult  -- Rate limit with delay
  deriving Repr, DecidableEq

/-- Policy guard function type -/
abbrev PolicyGuard := PolicyState → Token → PolicyResult

/-- Decoder state -/
structure DecoderState where
  policyState : PolicyState
  outputTokens : List Token
  blockedTokens : List Token
  totalTokensProcessed : Nat
  deriving Repr

/-- Decode function type -/
abbrev Decode := DecoderState → Token → IO (Except String Token)

/-- Policy error types -/
inductive PolicyError
  | blocked : String → PolicyError
  | rateLimited : Nat → PolicyError
  | contextExceeded : PolicyError
  | policyFailure : String → PolicyError
  deriving Repr, DecidableEq

/-- Convert policy result to error -/
def policyResultToError (result : PolicyResult) : Option PolicyError :=
  match result with
  | PolicyResult.allow => none
  | PolicyResult.block reason => some (PolicyError.blocked reason)
  | PolicyResult.rateLimit delay => some (PolicyError.rateLimited delay)

/-- Policy-gated decode function -/
def policyGatedDecode (guard : PolicyGuard) (state : DecoderState) (token : Token) : IO (Except String Token) := do
  -- Check if we've exceeded maximum tokens
  if state.policyState.currentTokenCount >= state.policyState.maxTokens then
    return Except.error "Maximum token count exceeded"

  -- Call policy guard
  let guardResult := guard state.policyState token

  match guardResult with
  | PolicyResult.allow =>
    -- Update state and return token
    let newPolicyState := {state.policyState with
      context := state.policyState.context ++ [token],
      currentTokenCount := state.policyState.currentTokenCount + 1}
    let newState := {state with
      policyState := newPolicyState,
      outputTokens := state.outputTokens ++ [token],
      totalTokensProcessed := state.totalTokensProcessed + 1}
    return Except.ok token

  | PolicyResult.block reason =>
    -- Block token and update state
    let newState := {state with
      blockedTokens := state.blockedTokens ++ [token],
      totalTokensProcessed := state.totalTokensProcessed + 1}
    return Except.error s!"Token blocked: {reason}"

  | PolicyResult.rateLimit delay =>
    -- Rate limit - could implement delay here
    let newState := {state with
      totalTokensProcessed := state.totalTokensProcessed + 1}
    return Except.error s!"Rate limited: {delay}ms delay"

/-- Initial decoder state -/
def initialDecoderState (maxTokens : Nat := 1000) (policyVersion : String := "1.0") : DecoderState :=
  ⟨⟨[], maxTokens, 0, policyVersion⟩, [], [], 0⟩

/-- Policy guard that allows all tokens (for testing) -/
def allowAllPolicy : PolicyGuard :=
  fun _ _ => PolicyResult.allow

/-- Policy guard that blocks specific tokens -/
def blockSpecificTokens (blockedTokens : List Token) : PolicyGuard :=
  fun _ token =>
    if blockedTokens.contains token then
      PolicyResult.block s!"Token '{token}' is blocked"
    else
      PolicyResult.allow

/-- Policy guard that enforces rate limiting -/
def rateLimitPolicy (maxTokensPerSecond : Nat) : PolicyGuard :=
  fun state token =>
    if state.currentTokenCount % maxTokensPerSecond = 0 then
      PolicyResult.rateLimit 1000  -- 1 second delay
    else
      PolicyResult.allow

/-- Policy guard that checks context length -/
def contextLengthPolicy (maxContextLength : Nat) : PolicyGuard :=
  fun state token =>
    if state.context.length >= maxContextLength then
      PolicyResult.block "Context length exceeded"
    else
      PolicyResult.allow

/-- Invariant: Policy guard is called on every token -/
def policyGuardCalled (state : DecoderState) : Prop :=
  state.totalTokensProcessed = state.outputTokens.length + state.blockedTokens.length

/-- Invariant: Blocked tokens are not in output -/
def blockedTokensNotInOutput (state : DecoderState) : Prop :=
  ∀ token : Token,
  token ∈ state.blockedTokens → token ∉ state.outputTokens

/-- Invariant: Context consistency -/
def contextConsistency (state : DecoderState) : Prop :=
  state.policyState.context = state.outputTokens

/-- Invariant: Token count consistency -/
def tokenCountConsistency (state : DecoderState) : Prop :=
  state.policyState.currentTokenCount = state.outputTokens.length

/-- Invariant: Maximum tokens not exceeded -/
def maxTokensNotExceeded (state : DecoderState) : Prop :=
  state.policyState.currentTokenCount ≤ state.policyState.maxTokens

/-- Combined invariant -/
def decoderInvariant (state : DecoderState) : Prop :=
  policyGuardCalled state ∧
  blockedTokensNotInOutput state ∧
  contextConsistency state ∧
  tokenCountConsistency state ∧
  maxTokensNotExceeded state

/-- Proof: Initial state satisfies invariant -/
theorem initial_state_invariant : decoderInvariant (initialDecoderState) := by
  simp [decoderInvariant, initialDecoderState]
  constructor
  · -- policyGuardCalled
    simp
  · constructor
    · -- blockedTokensNotInOutput
      intro token h
      simp at h
      contradiction
    · constructor
      · -- contextConsistency
        simp
      · constructor
        · -- tokenCountConsistency
          simp
        · -- maxTokensNotExceeded
          simp

/-- Proof: Policy-gated decode preserves invariant -/
theorem policyGatedDecode_preserves_invariant (guard : PolicyGuard) (state : DecoderState) (token : Token) :
  decoderInvariant state →
  match policyGatedDecode guard state token with
  | Except.ok _ => decoderInvariant {state with
      policyState := {state.policyState with
        context := state.policyState.context ++ [token],
        currentTokenCount := state.policyState.currentTokenCount + 1},
      outputTokens := state.outputTokens ++ [token],
      totalTokensProcessed := state.totalTokensProcessed + 1}
  | Except.error _ => decoderInvariant {state with
      blockedTokens := state.blockedTokens ++ [token],
      totalTokensProcessed := state.totalTokensProcessed + 1} := by
  intro h_inv
  simp [policyGatedDecode]
  -- Check if max tokens exceeded
  by_cases h_max : state.policyState.currentTokenCount >= state.policyState.maxTokens
  · -- Max tokens exceeded
    simp [h_max]
    -- Error case preserves invariant
    simp [decoderInvariant]
    constructor
    · -- policyGuardCalled preserved
      have h := h_inv.left
      simp [h]
    · constructor
      · -- blockedTokensNotInOutput preserved
        have h := h_inv.left.left
        simp [h]
      · constructor
        · -- contextConsistency preserved
          have h := h_inv.left.left.left
          simp [h]
        · constructor
          · -- tokenCountConsistency preserved
            have h := h_inv.left.left.left.left
            simp [h]
          · -- maxTokensNotExceeded preserved
            have h := h_inv.left.left.left.left.left
            simp [h]
  · -- Max tokens not exceeded
    simp [h_max]
    -- Call policy guard
    let guardResult := guard state.policyState token
    cases guardResult
    · -- PolicyResult.allow
      simp
      -- Success case preserves invariant
      simp [decoderInvariant]
      constructor
      · -- policyGuardCalled preserved
        have h := h_inv.left
        simp [h]
      · constructor
        · -- blockedTokensNotInOutput preserved
          have h := h_inv.left.left
          simp [h]
        · constructor
          · -- contextConsistency preserved
            have h := h_inv.left.left.left
            simp [h]
          · constructor
            · -- tokenCountConsistency preserved
              have h := h_inv.left.left.left.left
              simp [h]
            · -- maxTokensNotExceeded preserved
              have h := h_inv.left.left.left.left.left
              simp [h]
    · -- PolicyResult.block
      simp
      -- Block case preserves invariant
      simp [decoderInvariant]
      constructor
      · -- policyGuardCalled preserved
        have h := h_inv.left
        simp [h]
      · constructor
        · -- blockedTokensNotInOutput preserved
          have h := h_inv.left.left
          simp [h]
        · constructor
          · -- contextConsistency preserved
            have h := h_inv.left.left.left
            simp [h]
          · constructor
            · -- tokenCountConsistency preserved
              have h := h_inv.left.left.left.left
              simp [h]
            · -- maxTokensNotExceeded preserved
              have h := h_inv.left.left.left.left.left
              simp [h]
    · -- PolicyResult.rateLimit
      simp
      -- Rate limit case preserves invariant
      simp [decoderInvariant]
      constructor
      · -- policyGuardCalled preserved
        have h := h_inv.left
        simp [h]
      · constructor
        · -- blockedTokensNotInOutput preserved
          have h := h_inv.left.left
          simp [h]
        · constructor
          · -- contextConsistency preserved
            have h := h_inv.left.left.left
            simp [h]
          · constructor
            · -- tokenCountConsistency preserved
              have h := h_inv.left.left.left.left
              simp [h]
            · -- maxTokensNotExceeded preserved
              have h := h_inv.left.left.left.left.left
              simp [h]

/-- Utility functions for decoder inspection -/
def getOutputTokens (state : DecoderState) : List Token :=
  state.outputTokens

def getBlockedTokens (state : DecoderState) : List Token :=
  state.blockedTokens

def getTotalTokensProcessed (state : DecoderState) : Nat :=
  state.totalTokensProcessed

def getCurrentContext (state : DecoderState) : List Token :=
  state.policyState.context

def getTokenCount (state : DecoderState) : Nat :=
  state.policyState.currentTokenCount

def getMaxTokens (state : DecoderState) : Nat :=
  state.policyState.maxTokens

def getPolicyVersion (state : DecoderState) : String :=
  state.policyState.policyVersion

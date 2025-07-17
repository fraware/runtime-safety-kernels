/--
Policy extraction module for C kernel generation.

This module provides C-compatible interfaces for policy-gated decoding,
optimized for ultra-low latency safety checks with guaranteed enforcement.
-/

import RuntimeSafetyKernels.Policy
import Lean.Data.Json

/-- C-compatible policy configuration -/
structure CPolicyConfig where
  allowAllTokens : Bool
  blockedTokens : Array UInt32
  rateLimitTokensPerSecond : UInt32
  maxContextLength : UInt32
  maxTokensPerRequest : UInt32
  deriving Repr

/-- C-compatible policy guard result -/
structure CPolicyGuardResult where
  allowed : Bool
  blockedToken : Option UInt32
  rateLimited : Bool
  contextTooLong : Bool
  errorCode : UInt32
  deriving Repr

/-- C-compatible decoder state -/
structure CDecoderState where
  contextLength : UInt32
  tokensGenerated : UInt32
  lastTokenTime : UInt64
  tokensThisSecond : UInt32
  deriving Repr

/-- C-compatible decode result -/
structure CDecodeResult where
  success : Bool
  token : Option UInt32
  policyViolation : Bool
  errorCode : UInt32
  deriving Repr

/-- Convert Lean policy config to C config -/
def toCPolicyConfig (config : PolicyConfig) : CPolicyConfig :=
  ⟨config.allowAllTokens,
   config.blockedTokens.map (·.toUInt32),
   config.rateLimitTokensPerSecond.toUInt32,
   config.maxContextLength.toUInt32,
   config.maxTokensPerRequest.toUInt32⟩

/-- Convert C policy config to Lean config -/
def fromCPolicyConfig (config : CPolicyConfig) : PolicyConfig :=
  ⟨config.allowAllTokens,
   config.blockedTokens.map (·.toNat),
   config.rateLimitTokensPerSecond.toNat,
   config.maxContextLength.toNat,
   config.maxTokensPerRequest.toNat⟩

/-- Convert Lean policy guard result to C result -/
def toCPolicyGuardResult (result : PolicyGuardResult) : CPolicyGuardResult :=
  ⟨result.allowed,
   result.blockedToken.map (·.toUInt32),
   result.rateLimited,
   result.contextTooLong,
   result.errorCode.toUInt32⟩

/-- Convert C policy guard result to Lean result -/
def fromCPolicyGuardResult (result : CPolicyGuardResult) : PolicyGuardResult :=
  ⟨result.allowed,
   result.blockedToken.map (·.toNat),
   result.rateLimited,
   result.contextTooLong,
   result.errorCode.toNat⟩

/-- Convert Lean decoder state to C state -/
def toCDecoderState (state : DecoderState) : CDecoderState :=
  ⟨state.contextLength.toUInt32,
   state.tokensGenerated.toUInt32,
   state.lastTokenTime.toUInt64,
   state.tokensThisSecond.toUInt32⟩

/-- Convert C decoder state to Lean state -/
def fromCDecoderState (state : CDecoderState) : DecoderState :=
  ⟨state.contextLength.toNat,
   state.tokensGenerated.toNat,
   state.lastTokenTime.toNat,
   state.tokensThisSecond.toNat⟩

/-- Convert Lean decode result to C result -/
def toCDecodeResult (result : DecodeResult) : CDecodeResult :=
  ⟨result.success,
   result.token.map (·.toUInt32),
   result.policyViolation,
   result.errorCode.toUInt32⟩

/-- Convert C decode result to Lean result -/
def fromCDecodeResult (result : CDecodeResult) : DecodeResult :=
  ⟨result.success,
   result.token.map (·.toNat),
   result.policyViolation,
   result.errorCode.toNat⟩

/-- C-compatible policy manager -/
structure CPolicyManager where
  config : CPolicyConfig
  state : CDecoderState
  deriving Repr

/-- Create new C policy manager -/
def newCPolicyManager (config : CPolicyConfig) : CPolicyManager :=
  let initialState := CDecoderState.mk 0 0 0 0
  ⟨config, initialState⟩

/-- C-compatible policy guard function -/
def cPolicyGuard (manager : CPolicyManager) (token : UInt32) (currentTime : UInt64) : IO (CPolicyGuardResult × CPolicyManager) := do
  let leanConfig := fromCPolicyConfig manager.config
  let leanState := fromCDecoderState manager.state

  let result := policyGuard leanConfig leanState token.toNat currentTime.toNat
  let newState := result.snd

  let cResult := toCPolicyGuardResult result.fst
  let cNewState := toCDecoderState newState

  return (cResult, ⟨manager.config, cNewState⟩)

/-- C-compatible decode function -/
def cDecode (manager : CPolicyManager) (token : UInt32) (currentTime : UInt64) : IO (CDecodeResult × CPolicyManager) := do
  let leanConfig := fromCPolicyConfig manager.config
  let leanState := fromCDecoderState manager.state

  let result := decode leanConfig leanState token.toNat currentTime.toNat
  let newState := result.snd

  let cResult := toCDecodeResult result.fst
  let cNewState := toCDecoderState newState

  return (cResult, ⟨manager.config, cNewState⟩)

/-- C-compatible policy statistics -/
def cGetPolicyStats (manager : CPolicyManager) : IO (UInt32 × UInt32 × UInt32) := do
  let leanState := fromCDecoderState manager.state
  let stats := getPolicyStats leanState
  return (stats.fst.toUInt32, stats.snd.fst.toUInt32, stats.snd.snd.toUInt32)

/-- C-compatible health check -/
def cIsHealthy (manager : CPolicyManager) : Bool :=
  let leanConfig := fromCPolicyConfig manager.config
  let leanState := fromCDecoderState manager.state
  isHealthy leanConfig leanState

/-- C-compatible reset function -/
def cReset (manager : CPolicyManager) : CPolicyManager :=
  let resetState := CDecoderState.mk 0 0 0 0
  ⟨manager.config, resetState⟩

/-- Generate C header file -/
def generateCHeader : String :=
"#ifndef RSK_POLICY_H
#define RSK_POLICY_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern \"C\" {
#endif

// Policy configuration
typedef struct {
    bool allow_all_tokens;
    uint32_t* blocked_tokens;
    uint32_t blocked_tokens_count;
    uint32_t rate_limit_tokens_per_second;
    uint32_t max_context_length;
    uint32_t max_tokens_per_request;
} rsk_policy_config_t;

// Policy guard result
typedef struct {
    bool allowed;
    uint32_t blocked_token;
    bool blocked_token_present;
    bool rate_limited;
    bool context_too_long;
    uint32_t error_code;
} rsk_policy_guard_result_t;

// Decoder state
typedef struct {
    uint32_t context_length;
    uint32_t tokens_generated;
    uint64_t last_token_time;
    uint32_t tokens_this_second;
} rsk_decoder_state_t;

// Decode result
typedef struct {
    bool success;
    uint32_t token;
    bool token_present;
    bool policy_violation;
    uint32_t error_code;
} rsk_decode_result_t;

// Policy manager
typedef struct rsk_policy_manager rsk_policy_manager_t;

// Create new policy manager
rsk_policy_manager_t* rsk_policy_new(rsk_policy_config_t config);

// Policy guard function
rsk_policy_guard_result_t rsk_policy_guard(
    rsk_policy_manager_t* manager,
    uint32_t token,
    uint64_t current_time
);

// Decode function
rsk_decode_result_t rsk_decode(
    rsk_policy_manager_t* manager,
    uint32_t token,
    uint64_t current_time
);

// Get policy statistics
void rsk_get_policy_stats(
    rsk_policy_manager_t* manager,
    uint32_t* context_length,
    uint32_t* tokens_generated,
    uint32_t* tokens_this_second
);

// Health check
bool rsk_policy_is_healthy(rsk_policy_manager_t* manager);

// Reset policy manager
void rsk_policy_reset(rsk_policy_manager_t* manager);

// Free policy manager
void rsk_policy_free(rsk_policy_manager_t* manager);

#ifdef __cplusplus
}
#endif

#endif // RSK_POLICY_H"

/-- Main extraction entry point -/
def main : IO Unit := do
  -- Generate C header
  IO.FS.writeFile "src/extracted/rsk_policy.h" generateCHeader

  -- Run extraction tests
  let config := CPolicyConfig.mk false #[1, 2, 3] 100 1000 100
  let manager := newCPolicyManager config

  let (guardResult, newManager) ← cPolicyGuard manager 5 1000
  IO.println s!"Policy guard result: {guardResult}"

  let (decodeResult, finalManager) ← cDecode newManager 5 1000
  IO.println s!"Decode result: {decodeResult}"

  let (contextLen, tokensGen, tokensSec) ← cGetPolicyStats finalManager
  IO.println s!"Stats: context={contextLen}, generated={tokensGen}, this_second={tokensSec}"

  IO.println "Policy extraction completed successfully"

/-- Export for Lake build -/
#eval main

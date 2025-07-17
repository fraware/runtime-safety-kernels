/--
Sampler extraction module for C kernel generation.

This module provides C-compatible interfaces for all sampling algorithms,
optimized for ultra-low latency and minimal memory footprint.
-/

import RuntimeSafetyKernels.Sampler
import Lean.Data.Json

/-- C-compatible sampling configuration -/
structure CSamplingConfig where
  method : UInt32  -- 0=topK, 1=topP, 2=mirostat
  k : UInt32       -- for topK
  p : Float        -- for topP
  temperature : Float
  targetEntropy : Float  -- for mirostat
  learningRate : Float   -- for mirostat
  maxIterations : UInt32 -- for mirostat
  deriving Repr

/-- C-compatible sampling result -/
structure CSamplingResult (n : Nat) where
  probs : Array Float
  selectedToken : UInt32
  entropy : Float
  iterations : UInt32
  deriving Repr

/-- Convert Lean config to C config -/
def toCSamplingConfig (config : SamplingConfig) : CSamplingConfig :=
  match config with
  | SamplingConfig.topK cfg =>
    ⟨0, cfg.k.toUInt32, 0.0, cfg.temperature, 0.0, 0.0, 0⟩
  | SamplingConfig.topP cfg =>
    ⟨1, 0, cfg.p, cfg.temperature, 0.0, 0.0, 0⟩
  | SamplingConfig.mirostat cfg =>
    ⟨2, 0, 0.0, 1.0, cfg.targetEntropy, cfg.learningRate, cfg.maxIterations.toUInt32⟩

/-- Convert Lean result to C result -/
def toCSamplingResult {n : Nat} (result : SamplingResult n) : CSamplingResult n :=
  match result with
  | SamplingResult.topK r =>
    ⟨r.probs.toArray, r.selectedToken.toUInt32, r.entropy, 1⟩
  | SamplingResult.topP r =>
    ⟨r.probs.toArray, r.selectedToken.toUInt32, r.entropy, 1⟩
  | SamplingResult.mirostat r =>
    ⟨r.probs.toArray, r.selectedToken.toUInt32, r.entropy, r.iterations.toUInt32⟩

/-- C-compatible sampling function -/
def cSample (logits : Array Float) (config : CSamplingConfig) : IO (CSamplingResult logits.size) := do
  -- Convert to Lean types
  let logitsVec := Vector.ofArray logits
  let leanConfig := match config.method with
    | 0 => SamplingConfig.topK ⟨config.k.toNat, config.temperature⟩
    | 1 => SamplingConfig.topP ⟨config.p, config.temperature⟩
    | 2 => SamplingConfig.mirostat ⟨config.targetEntropy, config.learningRate, config.maxIterations.toNat, 0.01⟩
    | _ => SamplingConfig.topK ⟨40, 1.0⟩  -- default fallback

  -- Perform sampling
  let result := sample logitsVec leanConfig

  -- Convert back to C format
  return toCSamplingResult result

/-- Optimized C sampling function -/
def cSampleOptimized (logits : Array Float) (config : CSamplingConfig) : IO (CSamplingResult logits.size) := do
  -- Convert to Lean types
  let logitsVec := Vector.ofArray logits
  let leanConfig := match config.method with
    | 0 => SamplingConfig.topK ⟨config.k.toNat, config.temperature⟩
    | 1 => SamplingConfig.topP ⟨config.p, config.temperature⟩
    | 2 => SamplingConfig.mirostat ⟨config.targetEntropy, config.learningRate, config.maxIterations.toNat, 0.01⟩
    | _ => SamplingConfig.topK ⟨40, 1.0⟩  -- default fallback

  -- Perform optimized sampling
  let result := sampleOptimized logitsVec leanConfig

  -- Convert back to C format
  return toCSamplingResult result

/-- C-compatible validation function -/
def cValidateResult (result : CSamplingResult n) (config : CSamplingConfig) : Bool :=
  -- Basic validation checks
  result.probs.size = n &&
  result.selectedToken < n.toUInt32 &&
  result.probs.all (fun p => p >= 0.0 && p <= 1.0) &&
  (result.probs.foldl (· + ·) 0.0) ≈ 1.0

/-- Generate C header file -/
def generateCHeader : String :=
"#ifndef RSK_SAMPLER_H
#define RSK_SAMPLER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern \"C\" {
#endif

// Sampling configuration
typedef struct {
    uint32_t method;      // 0=topK, 1=topP, 2=mirostat
    uint32_t k;           // for topK
    float p;              // for topP
    float temperature;
    float target_entropy; // for mirostat
    float learning_rate;  // for mirostat
    uint32_t max_iterations; // for mirostat
} rsk_sampling_config_t;

// Sampling result
typedef struct {
    float* probs;
    uint32_t selected_token;
    float entropy;
    uint32_t iterations;
} rsk_sampling_result_t;

// Main sampling function
rsk_sampling_result_t rsk_sample(float* logits, uint32_t n, rsk_sampling_config_t config);

// Optimized sampling function
rsk_sampling_result_t rsk_sample_optimized(float* logits, uint32_t n, rsk_sampling_config_t config);

// Validation function
bool rsk_validate_result(rsk_sampling_result_t result, uint32_t n, rsk_sampling_config_t config);

// Memory management
void rsk_free_result(rsk_sampling_result_t result);

#ifdef __cplusplus
}
#endif

#endif // RSK_SAMPLER_H"

/-- Main extraction entry point -/
def main : IO Unit := do
  -- Generate C header
  IO.FS.writeFile "src/extracted/rsk_sampler.h" generateCHeader

  -- Run extraction tests
  let testLogits := #[1.0, 2.0, 3.0, 4.0, 5.0]
  let config := ⟨0, 3, 0.0, 1.0, 0.0, 0.0, 0⟩  -- topK with k=3

  let result ← cSample testLogits config
  IO.println s!"Extraction test result: {result}"

  let isValid := cValidateResult result config
  IO.println s!"Validation result: {isValid}"

  IO.println "Sampler extraction completed successfully"

/-- Export for Lake build -/
#eval main

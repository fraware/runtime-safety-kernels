/--
Main Sampler module providing unified interface for all RSK-1 sampling algorithms.

This module exports all mathematically-sound sampling algorithms with formal proofs:
- Core logits to probabilities conversion
- Top-K sampling with k-bounded proofs
- Top-P (nucleus) sampling with cumulative cutoff proofs
- Mirostat 2.0 with entropy error bounds
-/

import RuntimeSafetyKernels.Sampler.Core
import RuntimeSafetyKernels.Sampler.TopK
import RuntimeSafetyKernels.Sampler.TopP
import RuntimeSafetyKernels.Sampler.Mirostat

/-- Main Sampler module -/
module RuntimeSafetyKernels.Sampler

/-- Unified sampling configuration -/
inductive SamplingConfig
  | topK : TopKConfig → SamplingConfig
  | topP : TopPConfig → SamplingConfig
  | mirostat : MirostatConfig → SamplingConfig
  deriving Repr

/-- Unified sampling result -/
inductive SamplingResult (n : Nat)
  | topK : TopKResult n → SamplingResult n
  | topP : TopPResult n → SamplingResult n
  | mirostat : MirostatResult n → SamplingResult n
  deriving Repr

/-- Unified sampling function -/
def sample (logits : Logits n) (config : SamplingConfig) : SamplingResult n :=
  match config with
  | SamplingConfig.topK cfg => SamplingResult.topK (topKSample logits cfg)
  | SamplingConfig.topP cfg => SamplingResult.topP (topPSample logits cfg)
  | SamplingConfig.mirostat cfg => SamplingResult.mirostat (mirostatSample logits cfg)

/-- Optimized sampling function -/
def sampleOptimized (logits : Logits n) (config : SamplingConfig) : SamplingResult n :=
  match config with
  | SamplingConfig.topK cfg => SamplingResult.topK (topKSampleOptimized logits cfg)
  | SamplingConfig.topP cfg => SamplingResult.topP (topPSampleOptimized logits cfg)
  | SamplingConfig.mirostat cfg => SamplingResult.mirostat (mirostatSampleOptimized logits cfg)

/-- Extract probabilities from any sampling result -/
def getProbabilities {n : Nat} (result : SamplingResult n) : Vector Float n :=
  match result with
  | SamplingResult.topK r => r.probs
  | SamplingResult.topP r => r.probs
  | SamplingResult.mirostat r => r.probs

/-- Validate any sampling result -/
def isValidSamplingResult {n : Nat} (result : SamplingResult n) (config : SamplingConfig) : Bool :=
  match result, config with
  | SamplingResult.topK r, SamplingConfig.topK cfg => isValidTopKResult r
  | SamplingResult.topP r, SamplingConfig.topP cfg => isValidTopPResult r cfg.p
  | SamplingResult.mirostat r, SamplingConfig.mirostat cfg => isValidMirostatResult r cfg
  | _, _ => false

/-- Proof that unified sampling produces valid results -/
theorem sample_valid {n : Nat} (logits : Logits n) (config : SamplingConfig) :
  isValidSamplingResult (sample logits config) config := by
  cases config
  · -- Top-K case
    simp [sample, isValidSamplingResult]
    have h := topKSample_valid logits config
    simp [h]
  · -- Top-P case
    simp [sample, isValidSamplingResult]
    have h := topPSample_valid logits config
    simp [h]
  · -- Mirostat case
    simp [sample, isValidSamplingResult]
    have h := mirostatSample_valid logits config
    simp [h]

/-- Proof that optimized sampling preserves correctness -/
theorem sampleOptimized_correct {n : Nat} (logits : Logits n) (config : SamplingConfig) :
  let result := sample logits config
  let resultOpt := sampleOptimized logits config
  getProbabilities result = getProbabilities resultOpt := by
  cases config
  · -- Top-K case
    simp [sample, sampleOptimized, getProbabilities]
    have h := topKSampleOptimized_correct logits config
    simp [h]
  · -- Top-P case
    simp [sample, sampleOptimized, getProbabilities]
    have h := topPSampleOptimized_correct logits config
    simp [h]
  · -- Mirostat case
    simp [sample, sampleOptimized, getProbabilities]
    have h := mirostatSampleOptimized_correct logits config
    simp [h]

/-- Convenience constructors for common configurations -/
def mkTopK (k : Nat) (temperature : Float := 1.0) : SamplingConfig :=
  SamplingConfig.topK ⟨k, temperature⟩

def mkTopP (p : Float) (temperature : Float := 1.0) : SamplingConfig :=
  SamplingConfig.topP ⟨p, temperature⟩

def mkMirostat (targetEntropy : Float) (learningRate : Float := 0.1) (maxIterations : Nat := 100) : SamplingConfig :=
  SamplingConfig.mirostat ⟨targetEntropy, learningRate, maxIterations, 0.01⟩

/-- Performance benchmarks for sampling algorithms -/
def benchmarkSampling {n : Nat} (logits : Logits n) (iterations : Nat := 1000) : IO Unit := do
  let configs := [
    mkTopK 40,
    mkTopP 0.9,
    mkMirostat 3.0
  ]

  for config in configs do
    let start ← IO.monoMsNow
    for _ in List.range iterations do
      let _ := sample logits config
      pure ()
    let end ← IO.monoMsNow
    let duration := end - start
    IO.println s!"{config}: {duration}ms for {iterations} iterations"

/-- Monte Carlo validation for sampling correctness -/
def monteCarloValidation {n : Nat} (iterations : Nat := 1000000) : IO Bool := do
  let mutable allValid := true

  for _ in List.range iterations do
    -- Generate random logits
    let logits := Vector.generate n (fun _ => Float.random)

    -- Test all sampling methods
    let configs := [mkTopK 40, mkTopP 0.9, mkMirostat 3.0]

    for config in configs do
      let result := sample logits config
      if !isValidSamplingResult result config then
        allValid := false

  return allValid

/-- Export all core functionality -/
export RuntimeSafetyKernels.Sampler.Core
export RuntimeSafetyKernels.Sampler.TopK
export RuntimeSafetyKernels.Sampler.TopP
export RuntimeSafetyKernels.Sampler.Mirostat

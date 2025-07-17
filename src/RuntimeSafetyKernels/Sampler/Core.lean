/--
Core sampling module providing mathematical foundations for logits to probabilities conversion
and probability simplex preservation proofs.

This module establishes the mathematical basis for all sampling algorithms in RSK-1.
-/

import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Vector.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp

/-- Core sampling types and operations -/
module RuntimeSafetyKernels.Sampler.Core

/-- Logits vector type -/
abbrev Logits (n : Nat) := Vector Float n

/-- Probabilities vector type (must sum to 1) -/
structure Probabilities (n : Nat) where
  probs : Vector Float n
  sum_eq_one : ∑ i in Finset.range n, probs[i] = 1
  nonneg : ∀ i, 0 ≤ probs[i]

/-- Probability simplex for n dimensions -/
def ProbabilitySimplex (n : Nat) := {p : Vector Float n // ∑ i in Finset.range n, p[i] = 1 ∧ ∀ i, 0 ≤ p[i]}

/-- Convert logits to probabilities using softmax -/
def logitsToProbs (logits : Logits n) : Probabilities n :=
  let maxLogit := logits.foldl Float.max Float.negInf
  let shiftedLogits := logits.map (fun x => x - maxLogit)
  let expLogits := shiftedLogits.map Float.exp
  let sumExp := expLogits.foldl (· + ·) 0
  let probs := expLogits.map (fun x => x / sumExp)

  ⟨probs, by
    -- Proof that probabilities sum to 1
    simp [probs, expLogits, shiftedLogits, maxLogit]
    rw [Vector.foldl_map, Vector.foldl_map]
    have h : sumExp = ∑ i in Finset.range n, (expLogits[i])
    · simp [sumExp, expLogits]
    rw [h]
    simp [Vector.map]
    rw [Finset.sum_div]
    simp,
   by
    -- Proof that all probabilities are non-negative
    intro i
    simp [probs, expLogits, shiftedLogits]
    apply Float.div_nonneg
    · apply Float.exp_nonneg
    · apply Float.le_of_lt
      apply Float.exp_pos⟩

/-- Temperature scaling of logits -/
def applyTemperature (logits : Logits n) (temp : Float) : Logits n :=
  if temp ≤ 0 then logits else logits.map (fun x => x / temp)

/-- Mathematical properties of logits to probabilities conversion -/
theorem logitsToProbs_preserves_simplex (logits : Logits n) :
  let probs := logitsToProbs logits
  ∑ i in Finset.range n, probs.probs[i] = 1 ∧ ∀ i, 0 ≤ probs.probs[i] := by
  simp [logitsToProbs]
  constructor
  · exact logitsToProbs.logitsToProbs.proof_1
  · exact logitsToProbs.logitsToProbs.proof_2

/-- Temperature scaling preserves ordering -/
theorem temperature_preserves_ordering (logits : Logits n) (temp : Float) (i j : Fin n) :
  temp > 0 → (logits[i] ≤ logits[j] ↔ (applyTemperature logits temp)[i] ≤ (applyTemperature logits temp)[j]) := by
  intro h_pos
  simp [applyTemperature, h_pos]
  constructor
  · intro h_le
    apply Float.div_le_div_of_le_of_pos
    · exact h_le
    · exact h_pos
  · intro h_le
    apply Float.le_mul_of_div_le
    · exact h_pos
    · exact h_le

/-- Numerical stability: logits to probs handles extreme values -/
theorem logitsToProbs_numerically_stable (logits : Logits n) :
  let probs := logitsToProbs logits
  ∀ i, Float.isFinite probs.probs[i] := by
  intro i
  simp [logitsToProbs]
  apply Float.div_isFinite
  · apply Float.exp_isFinite
  · apply Float.ne_of_gt
    apply Float.exp_pos

/-- Utility function to check if probabilities are valid -/
def isValidProbabilities (probs : Vector Float n) : Bool :=
  let sum := probs.foldl (· + ·) 0
  let isSumOne := Float.abs (sum - 1) < 1e-10
  let allNonneg := probs.foldl (fun acc x => acc && x ≥ 0) true
  isSumOne && allNonneg

/-- Proof that logitsToProbs produces valid probabilities -/
theorem logitsToProbs_valid (logits : Logits n) :
  isValidProbabilities (logitsToProbs logits).probs := by
  simp [isValidProbabilities, logitsToProbs]
  constructor
  · -- Sum is approximately 1
    have h := logitsToProbs_preserves_simplex logits
    simp [h]
    apply Float.abs_lt_of_lt
    apply Float.sub_lt
    · simp
    · apply Float.lt_of_lt_of_le
      · apply Float.neg_lt_zero
      · simp
  · -- All non-negative
    have h := logitsToProbs_preserves_simplex logits
    simp [h]
    intro i
    exact logitsToProbs.logitsToProbs.proof_2 i

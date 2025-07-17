/--
Shape-safe tensor API specification for RSK-4.

This module defines the formal specification for shape-safe tensor operations that
prevent shape mismatches at compile time using dataset-safety-specs theorems.
-/

import Mathlib.Data.Vector.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Logic.Basic

/-- Shape specification module -/
module RuntimeSafetyKernels.Shape.Spec

/-- Tensor dimension type -/
abbrev Dimension := Nat

/-- Tensor shape (list of dimensions) -/
abbrev Shape := List Dimension

/-- Tensor data type -/
abbrev TensorData (shape : Shape) := Vector Float (shape.foldl (· * ·) 1)

/-- Tensor type with shape information -/
structure Tensor (shape : Shape) where
  data : TensorData shape
  deriving Repr

/-- Shape validation -/
def isValidShape (shape : Shape) : Bool :=
  shape.all (fun dim => dim > 0)

/-- Shape equality -/
def shapeEqual (shape1 shape2 : Shape) : Bool :=
  shape1 = shape2

/-- Shape compatibility for element-wise operations -/
def shapesCompatible (shape1 shape2 : Shape) : Bool :=
  shape1 = shape2

/-- Shape compatibility for matrix multiplication -/
def shapesCompatibleForMatMul (shape1 shape2 : Shape) : Bool :=
  match shape1, shape2 with
  | [m, n], [n', p] => n = n'
  | _, _ => false

/-- Shape compatibility for broadcasting -/
def shapesCompatibleForBroadcast (shape1 shape2 : Shape) : Bool :=
  let len1 := shape1.length
  let len2 := shape2.length
  let maxLen := max len1 len2

  let paddedShape1 := shape1 ++ List.replicate (maxLen - len1) 1
  let paddedShape2 := shape2 ++ List.replicate (maxLen - len2) 1

  List.zipWith (fun dim1 dim2 => dim1 = dim2 || dim1 = 1 || dim2 = 1) paddedShape1 paddedShape2
  |> List.all id

/-- Result shape for broadcasting -/
def broadcastShape (shape1 shape2 : Shape) : Option Shape :=
  if shapesCompatibleForBroadcast shape1 shape2 then
    let len1 := shape1.length
    let len2 := shape2.length
    let maxLen := max len1 len2

    let paddedShape1 := shape1 ++ List.replicate (maxLen - len1) 1
    let paddedShape2 := shape2 ++ List.replicate (maxLen - len2) 1

    let resultShape := List.zipWith max paddedShape1 paddedShape2
    some resultShape
  else
    none

/-- Result shape for matrix multiplication -/
def matMulShape (shape1 shape2 : Shape) : Option Shape :=
  if shapesCompatibleForMatMul shape1 shape2 then
    match shape1, shape2 with
    | [m, n], [n', p] => some [m, p]
    | _, _ => none
  else
    none

/-- Create tensor from data -/
def mkTensor (shape : Shape) (data : List Float) : Option (Tensor shape) :=
  if isValidShape shape ∧ data.length = shape.foldl (· * ·) 1 then
    some ⟨Vector.ofList data⟩
  else
    none

/-- Create zero tensor -/
def zeroTensor (shape : Shape) : Option (Tensor shape) :=
  if isValidShape shape then
    let size := shape.foldl (· * ·) 1
    let data := Vector.replicate size 0.0
    some ⟨data⟩
  else
    none

/-- Create identity matrix -/
def identityMatrix (size : Nat) : Option (Tensor [size, size]) :=
  if size > 0 then
    let data := List.range (size * size) |>.map (fun i =>
      if i % size = i / size then 1.0 else 0.0)
    mkTensor [size, size] data
  else
    none

/-- Element-wise addition -/
def tensorAdd {shape : Shape} (t1 t2 : Tensor shape) : Tensor shape :=
  ⟨t1.data.zipWith t2.data (· + ·)⟩

/-- Element-wise multiplication -/
def tensorMul {shape : Shape} (t1 t2 : Tensor shape) : Tensor shape :=
  ⟨t1.data.zipWith t2.data (· * ·)⟩

/-- Scalar multiplication -/
def scalarMul {shape : Shape} (t : Tensor shape) (scalar : Float) : Tensor shape :=
  ⟨t.data.map (fun x => x * scalar)⟩

/-- Matrix multiplication (with shape checking) -/
def tensorMatMul {shape1 shape2 : Shape} (t1 : Tensor shape1) (t2 : Tensor shape2) : Option (Tensor (matMulShape shape1 shape2).getD []) :=
  match matMulShape shape1 shape2 with
  | some resultShape =>
    match shape1, shape2 with
    | [m, n], [n', p] =>
      let resultData := List.range (m * p) |>.map (fun i =>
        let row := i / p
        let col := i % p
        List.range n |>.foldl (fun acc k =>
          acc + t1.data[row * n + k] * t2.data[k * p + col]) 0.0)
      mkTensor resultShape resultData
    | _, _ => none
  | none => none

/-- Tensor reshape -/
def tensorReshape {shape1 : Shape} (t : Tensor shape1) (shape2 : Shape) : Option (Tensor shape2) :=
  if isValidShape shape2 ∧ shape1.foldl (· * ·) 1 = shape2.foldl (· * ·) 1 then
    some ⟨t.data⟩
  else
    none

/-- Tensor transpose -/
def tensorTranspose {shape : Shape} (t : Tensor shape) : Option (Tensor shape.reverse) :=
  if shape.length = 2 then
    match shape with
    | [m, n] =>
      let transposedData := List.range (m * n) |>.map (fun i =>
        let row := i / n
        let col := i % n
        t.data[col * m + row])
      mkTensor [n, m] transposedData
    | _ => none
  else
    none

/-- Tensor slice -/
def tensorSlice {shape : Shape} (t : Tensor shape) (start : Nat) (length : Nat) : Option (Tensor [length]) :=
  if start + length ≤ shape.foldl (· * ·) 1 then
    let slicedData := List.range length |>.map (fun i => t.data[start + i])
    mkTensor [length] slicedData
  else
    none

/-- Tensor concatenation -/
def tensorConcat {shape1 shape2 : Shape} (t1 : Tensor shape1) (t2 : Tensor shape2) (axis : Nat) : Option (Tensor (tensorConcatShape shape1 shape2 axis)) :=
  if axis < shape1.length ∧ axis < shape2.length ∧
     List.zipWith (fun dim1 dim2 => dim1 = dim2) shape1 shape2 |>.all id then
    let newShape := List.zipWith (fun dim1 dim2 =>
      if axis = 0 then dim1 + dim2 else dim1) shape1 shape2
    let totalSize := newShape.foldl (· * ·) 1
    let data := t1.data.toList ++ t2.data.toList
    mkTensor newShape data
  else
    none

/-- Helper function for concatenation shape -/
def tensorConcatShape (shape1 shape2 : Shape) (axis : Nat) : Shape :=
  List.zipWith (fun dim1 dim2 =>
    if axis = 0 then dim1 + dim2 else dim1) shape1 shape2

/-- Invariant: Tensor data size matches shape -/
def tensorDataSizeMatchesShape {shape : Shape} (t : Tensor shape) : Prop :=
  t.data.length = shape.foldl (· * ·) 1

/-- Invariant: All tensor dimensions are positive -/
def tensorDimensionsPositive {shape : Shape} (t : Tensor shape) : Prop :=
  shape.all (fun dim => dim > 0)

/-- Invariant: Tensor data contains finite values -/
def tensorDataFinite {shape : Shape} (t : Tensor shape) : Prop :=
  t.data.all Float.isFinite

/-- Combined tensor invariant -/
def tensorInvariant {shape : Shape} (t : Tensor shape) : Prop :=
  tensorDataSizeMatchesShape t ∧
  tensorDimensionsPositive t ∧
  tensorDataFinite t

/-- Proof: Zero tensor satisfies invariant -/
theorem zero_tensor_invariant (shape : Shape) :
  isValidShape shape →
  match zeroTensor shape with
  | some t => tensorInvariant t
  | none => True := by
  intro h_valid
  simp [zeroTensor, tensorInvariant, tensorDataSizeMatchesShape, tensorDimensionsPositive, tensorDataFinite]
  by_cases h : isValidShape shape
  · -- Valid shape case
    simp [h]
    constructor
    · -- Data size matches shape
      simp [Vector.replicate]
    · constructor
      · -- Dimensions positive
        exact h
      · -- Data finite
        simp [Vector.replicate]
        intro i
        simp [Float.isFinite]
  · -- Invalid shape case
    simp [h]

/-- Proof: Tensor addition preserves invariant -/
theorem tensor_add_preserves_invariant {shape : Shape} (t1 t2 : Tensor shape) :
  tensorInvariant t1 → tensorInvariant t2 → tensorInvariant (tensorAdd t1 t2) := by
  intro h1 h2
  simp [tensorAdd, tensorInvariant, tensorDataSizeMatchesShape, tensorDimensionsPositive, tensorDataFinite]
  constructor
  · -- Data size preserved
    have h := h1.left
    simp [h, Vector.zipWith]
  · constructor
    · -- Dimensions positive preserved
      have h := h1.left.left
      exact h
    · -- Data finite preserved
      have h := h1.left.left.left
      simp [h, Vector.zipWith]
      intro i
      apply Float.add_isFinite
      · exact h i
      · have h2_finite := h2.left.left.left
        exact h2_finite i

/-- Proof: Scalar multiplication preserves invariant -/
theorem scalar_mul_preserves_invariant {shape : Shape} (t : Tensor shape) (scalar : Float) :
  tensorInvariant t → tensorInvariant (scalarMul t scalar) := by
  intro h
  simp [scalarMul, tensorInvariant, tensorDataSizeMatchesShape, tensorDimensionsPositive, tensorDataFinite]
  constructor
  · -- Data size preserved
    have h_size := h.left
    simp [h_size, Vector.map]
  · constructor
    · -- Dimensions positive preserved
      have h_dims := h.left.left
      exact h_dims
    · -- Data finite preserved
      have h_finite := h.left.left.left
      simp [h_finite, Vector.map]
      intro i
      apply Float.mul_isFinite
      · exact h_finite i
      · apply Float.isFinite_of_le
        apply Float.le_max

/-- Proof: Matrix multiplication preserves shape compatibility -/
theorem matmul_shape_compatibility {shape1 shape2 : Shape} (t1 : Tensor shape1) (t2 : Tensor shape2) :
  tensorInvariant t1 → tensorInvariant t2 →
  match tensorMatMul t1 t2 with
  | some result => tensorInvariant result
  | none => True := by
  intro h1 h2
  simp [tensorMatMul, tensorInvariant]
  -- Check shape compatibility
  by_cases h_compat : shapesCompatibleForMatMul shape1 shape2
  · -- Compatible shapes
    simp [h_compat]
    match shape1, shape2 with
    | [m, n], [n', p] =>
      simp
      -- Result satisfies invariant
      constructor
      · -- Data size matches shape
        simp [matMulShape, h_compat]
      · constructor
        · -- Dimensions positive
          simp [matMulShape, h_compat]
        · -- Data finite
          simp [matMulShape, h_compat]
  · -- Incompatible shapes
    simp [h_compat]

/-- Utility functions for tensor inspection -/
def getTensorShape {shape : Shape} (t : Tensor shape) : Shape :=
  shape

def getTensorSize {shape : Shape} (t : Tensor shape) : Nat :=
  shape.foldl (· * ·) 1

def getTensorData {shape : Shape} (t : Tensor shape) : TensorData shape :=
  t.data

def getTensorElement {shape : Shape} (t : Tensor shape) (index : Nat) : Option Float :=
  if index < getTensorSize t then
    some t.data[index]
  else
    none

def setTensorElement {shape : Shape} (t : Tensor shape) (index : Nat) (value : Float) : Option (Tensor shape) :=
  if index < getTensorSize t then
    let newData := t.data.set index value
    some ⟨newData⟩
  else
    none

/-- Shape validation utilities -/
def validateShape (shape : Shape) : Bool :=
  isValidShape shape

def validateTensor {shape : Shape} (t : Tensor shape) : Bool :=
  tensorInvariant t

def validateShapesCompatible (shape1 shape2 : Shape) : Bool :=
  shapesCompatible shape1 shape2

def validateShapesCompatibleForMatMul (shape1 shape2 : Shape) : Bool :=
  shapesCompatibleForMatMul shape1 shape2

def validateShapesCompatibleForBroadcast (shape1 shape2 : Shape) : Bool :=
  shapesCompatibleForBroadcast shape1 shape2

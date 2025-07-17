/--
Shape extraction module for C kernel generation.

This module provides C-compatible interfaces for shape-safe tensor operations,
optimized for ultra-low latency with guaranteed shape safety.
-/

import RuntimeSafetyKernels.Shape
import Lean.Data.Json

/-- C-compatible tensor shape -/
structure CTensorShape where
  dimensions : Array UInt32
  size : UInt32
  deriving Repr

/-- C-compatible tensor data -/
structure CTensorData where
  data : Array Float
  shape : CTensorShape
  deriving Repr

/-- C-compatible tensor operation result -/
structure CTensorResult where
  success : Bool
  data : Option CTensorData
  errorCode : UInt32
  errorMessage : String
  deriving Repr

/-- Convert Lean tensor shape to C shape -/
def toCTensorShape (shape : TensorShape) : CTensorShape :=
  ⟨shape.dimensions.map (·.toUInt32), shape.size.toUInt32⟩

/-- Convert C tensor shape to Lean shape -/
def fromCTensorShape (shape : CTensorShape) : TensorShape :=
  ⟨shape.dimensions.map (·.toNat), shape.size.toNat⟩

/-- Convert Lean tensor data to C data -/
def toCTensorData (tensor : TensorData) : CTensorData :=
  ⟨tensor.data.toArray, toCTensorShape tensor.shape⟩

/-- Convert C tensor data to Lean data -/
def fromCTensorData (tensor : CTensorData) : TensorData :=
  ⟨Vector.ofArray tensor.data, fromCTensorShape tensor.shape⟩

/-- Convert Lean tensor result to C result -/
def toCTensorResult (result : TensorResult) : CTensorResult :=
  match result with
  | TensorResult.success data =>
    ⟨true, some (toCTensorData data), 0, ""⟩
  | TensorResult.failure errorCode errorMessage =>
    ⟨false, none, errorCode.toUInt32, errorMessage⟩

/-- Convert C tensor result to Lean result -/
def fromCTensorResult (result : CTensorResult) : TensorResult :=
  if result.success then
    match result.data with
    | none => TensorResult.failure 1 "No data in successful result"
    | some data => TensorResult.success (fromCTensorData data)
  else
    TensorResult.failure result.errorCode.toNat result.errorMessage

/-- C-compatible tensor manager -/
structure CTensorManager where
  -- No state needed for pure tensor operations
  deriving Repr

/-- Create new C tensor manager -/
def newCTensorManager : CTensorManager :=
  ⟨⟩

/-- C-compatible tensor creation -/
def cCreateTensor (manager : CTensorManager) (data : Array Float) (shape : CTensorShape) : IO CTensorResult := do
  let leanData := Vector.ofArray data
  let leanShape := fromCTensorShape shape

  let result := createTensor leanData leanShape
  return toCTensorResult result

/-- C-compatible zero tensor creation -/
def cZeroTensor (manager : CTensorManager) (shape : CTensorShape) : IO CTensorResult := do
  let leanShape := fromCTensorShape shape

  let result := zeroTensor leanShape
  return toCTensorResult result

/-- C-compatible identity matrix creation -/
def cIdentityMatrix (manager : CTensorManager) (size : UInt32) : IO CTensorResult := do
  let result := identityMatrix size.toNat
  return toCTensorResult result

/-- C-compatible element-wise addition -/
def cAdd (manager : CTensorManager) (a : CTensorData) (b : CTensorData) : IO CTensorResult := do
  let leanA := fromCTensorData a
  let leanB := fromCTensorData b

  let result := add leanA leanB
  return toCTensorResult result

/-- C-compatible element-wise multiplication -/
def cMultiply (manager : CTensorManager) (a : CTensorData) (b : CTensorData) : IO CTensorResult := do
  let leanA := fromCTensorData a
  let leanB := fromCTensorData b

  let result := multiply leanA leanB
  return toCTensorResult result

/-- C-compatible scalar multiplication -/
def cScalarMultiply (manager : CTensorManager) (tensor : CTensorData) (scalar : Float) : IO CTensorResult := do
  let leanTensor := fromCTensorData tensor

  let result := scalarMultiply leanTensor scalar
  return toCTensorResult result

/-- C-compatible matrix multiplication -/
def cMatrixMultiply (manager : CTensorManager) (a : CTensorData) (b : CTensorData) : IO CTensorResult := do
  let leanA := fromCTensorData a
  let leanB := fromCTensorData b

  let result := matrixMultiply leanA leanB
  return toCTensorResult result

/-- C-compatible tensor reshape -/
def cReshape (manager : CTensorManager) (tensor : CTensorData) (newShape : CTensorShape) : IO CTensorResult := do
  let leanTensor := fromCTensorData tensor
  let leanNewShape := fromCTensorShape newShape

  let result := reshape leanTensor leanNewShape
  return toCTensorResult result

/-- C-compatible tensor transpose -/
def cTranspose (manager : CTensorManager) (tensor : CTensorData) : IO CTensorResult := do
  let leanTensor := fromCTensorData tensor

  let result := transpose leanTensor
  return toCTensorResult result

/-- C-compatible tensor slice -/
def cSlice (manager : CTensorManager) (tensor : CTensorData) (start : Array UInt32) (end : Array UInt32) : IO CTensorResult := do
  let leanTensor := fromCTensorData tensor
  let leanStart := start.map (·.toNat)
  let leanEnd := end.map (·.toNat)

  let result := slice leanTensor leanStart leanEnd
  return toCTensorResult result

/-- C-compatible tensor concatenation -/
def cConcatenate (manager : CTensorManager) (tensors : Array CTensorData) (axis : UInt32) : IO CTensorResult := do
  let leanTensors := tensors.map fromCTensorData

  let result := concatenate leanTensors axis.toNat
  return toCTensorResult result

/-- C-compatible shape validation -/
def cValidateShape (manager : CTensorManager) (shape : CTensorShape) : Bool :=
  let leanShape := fromCTensorShape shape
  validateShape leanShape

/-- C-compatible tensor validation -/
def cValidateTensor (manager : CTensorManager) (tensor : CTensorData) : Bool :=
  let leanTensor := fromCTensorData tensor
  validateTensor leanTensor

/-- Generate C header file -/
def generateCHeader : String :=
"#ifndef RSK_SHAPE_H
#define RSK_SHAPE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern \"C\" {
#endif

// Tensor shape
typedef struct {
    uint32_t* dimensions;
    uint32_t dimensions_count;
    uint32_t size;
} rsk_tensor_shape_t;

// Tensor data
typedef struct {
    float* data;
    uint32_t data_count;
    rsk_tensor_shape_t shape;
} rsk_tensor_data_t;

// Tensor operation result
typedef struct {
    bool success;
    rsk_tensor_data_t data;
    uint32_t error_code;
    char* error_message;
} rsk_tensor_result_t;

// Tensor manager
typedef struct rsk_tensor_manager rsk_tensor_manager_t;

// Create new tensor manager
rsk_tensor_manager_t* rsk_tensor_new(void);

// Tensor creation
rsk_tensor_result_t rsk_create_tensor(
    rsk_tensor_manager_t* manager,
    float* data,
    uint32_t data_count,
    rsk_tensor_shape_t shape
);

// Zero tensor creation
rsk_tensor_result_t rsk_zero_tensor(
    rsk_tensor_manager_t* manager,
    rsk_tensor_shape_t shape
);

// Identity matrix creation
rsk_tensor_result_t rsk_identity_matrix(
    rsk_tensor_manager_t* manager,
    uint32_t size
);

// Element-wise addition
rsk_tensor_result_t rsk_add(
    rsk_tensor_manager_t* manager,
    rsk_tensor_data_t a,
    rsk_tensor_data_t b
);

// Element-wise multiplication
rsk_tensor_result_t rsk_multiply(
    rsk_tensor_manager_t* manager,
    rsk_tensor_data_t a,
    rsk_tensor_data_t b
);

// Scalar multiplication
rsk_tensor_result_t rsk_scalar_multiply(
    rsk_tensor_manager_t* manager,
    rsk_tensor_data_t tensor,
    float scalar
);

// Matrix multiplication
rsk_tensor_result_t rsk_matrix_multiply(
    rsk_tensor_manager_t* manager,
    rsk_tensor_data_t a,
    rsk_tensor_data_t b
);

// Tensor reshape
rsk_tensor_result_t rsk_reshape(
    rsk_tensor_manager_t* manager,
    rsk_tensor_data_t tensor,
    rsk_tensor_shape_t new_shape
);

// Tensor transpose
rsk_tensor_result_t rsk_transpose(
    rsk_tensor_manager_t* manager,
    rsk_tensor_data_t tensor
);

// Tensor slice
rsk_tensor_result_t rsk_slice(
    rsk_tensor_manager_t* manager,
    rsk_tensor_data_t tensor,
    uint32_t* start,
    uint32_t start_count,
    uint32_t* end,
    uint32_t end_count
);

// Tensor concatenation
rsk_tensor_result_t rsk_concatenate(
    rsk_tensor_manager_t* manager,
    rsk_tensor_data_t* tensors,
    uint32_t tensors_count,
    uint32_t axis
);

// Shape validation
bool rsk_validate_shape(rsk_tensor_manager_t* manager, rsk_tensor_shape_t shape);

// Tensor validation
bool rsk_validate_tensor(rsk_tensor_manager_t* manager, rsk_tensor_data_t tensor);

// Free tensor result
void rsk_free_tensor_result(rsk_tensor_result_t result);

// Free tensor manager
void rsk_tensor_free(rsk_tensor_manager_t* manager);

#ifdef __cplusplus
}
#endif

#endif // RSK_SHAPE_H"

/-- Generate GPT-2 compilation test -/
def generateGPT2Test : String :=
"#include \"rsk_shape.h\"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Create tensor manager
    rsk_tensor_manager_t* manager = rsk_tensor_new();
    if (!manager) {
        printf(\"Failed to create tensor manager\\n\");
        return 1;
    }

    // Test GPT-2 attention computation
    // Create query, key, value tensors (batch_size=1, seq_len=512, hidden_size=768)
    uint32_t hidden_size = 768;
    uint32_t seq_len = 512;
    uint32_t batch_size = 1;

    rsk_tensor_shape_t qkv_shape = {
        .dimensions = (uint32_t[]){batch_size, seq_len, hidden_size},
        .dimensions_count = 3,
        .size = batch_size * seq_len * hidden_size
    };

    // Create random data for Q, K, V
    float* q_data = malloc(qkv_shape.size * sizeof(float));
    float* k_data = malloc(qkv_shape.size * sizeof(float));
    float* v_data = malloc(qkv_shape.size * sizeof(float));

    for (uint32_t i = 0; i < qkv_shape.size; i++) {
        q_data[i] = (float)rand() / RAND_MAX;
        k_data[i] = (float)rand() / RAND_MAX;
        v_data[i] = (float)rand() / RAND_MAX;
    }

    rsk_tensor_data_t q_tensor = {.data = q_data, .data_count = qkv_shape.size, .shape = qkv_shape};
    rsk_tensor_data_t k_tensor = {.data = k_data, .data_count = qkv_shape.size, .shape = qkv_shape};
    rsk_tensor_data_t v_tensor = {.data = v_data, .data_count = qkv_shape.size, .shape = qkv_shape};

    // Compute attention scores: Q * K^T
    rsk_tensor_result_t k_transposed = rsk_transpose(manager, k_tensor);
    if (!k_transposed.success) {
        printf(\"Failed to transpose K tensor\\n\");
        return 1;
    }

    rsk_tensor_result_t attention_scores = rsk_matrix_multiply(manager, q_tensor, k_transposed.data);
    if (!attention_scores.success) {
        printf(\"Failed to compute attention scores\\n\");
        return 1;
    }

    // Apply softmax (simplified - just normalize)
    float max_score = attention_scores.data.data[0];
    for (uint32_t i = 1; i < attention_scores.data.data_count; i++) {
        if (attention_scores.data.data[i] > max_score) {
            max_score = attention_scores.data.data[i];
        }
    }

    for (uint32_t i = 0; i < attention_scores.data.data_count; i++) {
        attention_scores.data.data[i] = (attention_scores.data.data[i] - max_score);
    }

    // Compute attention output: attention_scores * V
    rsk_tensor_result_t attention_output = rsk_matrix_multiply(manager, attention_scores.data, v_tensor);
    if (!attention_output.success) {
        printf(\"Failed to compute attention output\\n\");
        return 1;
    }

    printf(\"GPT-2 attention computation successful!\\n\");
    printf(\"Output tensor shape: %u x %u x %u\\n\",
           attention_output.data.shape.dimensions[0],
           attention_output.data.shape.dimensions[1],
           attention_output.data.shape.dimensions[2]);

    // Cleanup
    rsk_free_tensor_result(k_transposed);
    rsk_free_tensor_result(attention_scores);
    rsk_free_tensor_result(attention_output);
    rsk_tensor_free(manager);

    free(q_data);
    free(k_data);
    free(v_data);

    return 0;
}"

/-- Main extraction entry point -/
def main : IO Unit := do
  -- Generate C header
  IO.FS.writeFile "src/extracted/rsk_shape.h" generateCHeader

  -- Generate GPT-2 test
  IO.FS.writeFile "src/extracted/gpt2_test.c" generateGPT2Test

  -- Run extraction tests
  let manager := newCTensorManager

  -- Test tensor creation
  let shape := CTensorShape.mk #[2, 3] 6
  let data := #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

  let result ← cCreateTensor manager data shape
  IO.println s!"Tensor creation result: {result}"

  -- Test matrix multiplication
  let a := CTensorData.mk #[1.0, 2.0, 3.0, 4.0] (CTensorShape.mk #[2, 2] 4)
  let b := CTensorData.mk #[5.0, 6.0, 7.0, 8.0] (CTensorShape.mk #[2, 2] 4)

  let multResult ← cMatrixMultiply manager a b
  IO.println s!"Matrix multiplication result: {multResult}"

  IO.println "Shape extraction completed successfully"

/-- Export for Lake build -/
#eval main

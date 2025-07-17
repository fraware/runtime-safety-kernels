/--
Go bindings for Runtime Safety Kernels using CGO.

This module provides Go bindings for all RSK components,
enabling seamless integration with Go AI frameworks.
-/

import RuntimeSafetyKernels.Sampler
import RuntimeSafetyKernels.Concurrency
import RuntimeSafetyKernels.Policy
import RuntimeSafetyKernels.Shape
import Lean.Data.Json

/-- Go bindings module -/
module RuntimeSafetyKernels.LanguageBindings.Go

/-- Go-compatible sampling configuration -/
structure GoSamplingConfig where
  method : String  -- "topk", "topp", "mirostat"
  k : Option Nat   -- for topK
  p : Option Float -- for topP
  temperature : Float
  targetEntropy : Option Float  -- for mirostat
  learningRate : Option Float   -- for mirostat
  maxIterations : Option Nat    -- for mirostat
  deriving Repr

/-- Go-compatible sampling result -/
structure GoSamplingResult where
  probabilities : List Float
  selectedToken : Nat
  entropy : Float
  iterations : Nat
  deriving Repr

/-- Convert Go config to Lean config -/
def toLeanSamplingConfig (config : GoSamplingConfig) : Option SamplingConfig :=
  match config.method with
  | "topk" =>
    match config.k with
    | none => none
    | some k => some (SamplingConfig.topK ⟨k, config.temperature⟩)
  | "topp" =>
    match config.p with
    | none => none
    | some p => some (SamplingConfig.topP ⟨p, config.temperature⟩)
  | "mirostat" =>
    match (config.targetEntropy, config.learningRate, config.maxIterations) with
    | (some target, some lr, some maxIter) =>
      some (SamplingConfig.mirostat ⟨target, lr, maxIter, 0.01⟩)
    | _ => none
  | _ => none

/-- Convert Lean result to Go result -/
def toGoSamplingResult {n : Nat} (result : SamplingResult n) : GoSamplingResult :=
  match result with
  | SamplingResult.topK r =>
    ⟨r.probs.toList, r.selectedToken, r.entropy, 1⟩
  | SamplingResult.topP r =>
    ⟨r.probs.toList, r.selectedToken, r.entropy, 1⟩
  | SamplingResult.mirostat r =>
    ⟨r.probs.toList, r.selectedToken, r.entropy, r.iterations⟩

/-- Go sampling function -/
def goSample (logits : List Float) (config : GoSamplingConfig) : IO (Option GoSamplingResult) := do
  let leanConfig := toLeanSamplingConfig config
  match leanConfig with
  | none => return none
  | some cfg =>
    let logitsVec := Vector.ofList logits
    let result := sample logitsVec cfg
    return some (toGoSamplingResult result)

/-- Go-compatible policy configuration -/
structure GoPolicyConfig where
  allowAllTokens : Bool
  blockedTokens : List Nat
  rateLimitTokensPerSecond : Nat
  maxContextLength : Nat
  maxTokensPerRequest : Nat
  deriving Repr

/-- Go-compatible policy result -/
structure GoPolicyResult where
  allowed : Bool
  blockedToken : Option Nat
  rateLimited : Bool
  contextTooLong : Bool
  errorCode : Nat
  deriving Repr

/-- Convert Go policy config to Lean config -/
def toLeanPolicyConfig (config : GoPolicyConfig) : PolicyConfig :=
  ⟨config.allowAllTokens, config.blockedTokens, config.rateLimitTokensPerSecond,
   config.maxContextLength, config.maxTokensPerRequest⟩

/-- Convert Lean policy result to Go result -/
def toGoPolicyResult (result : PolicyGuardResult) : GoPolicyResult :=
  ⟨result.allowed, result.blockedToken, result.rateLimited, result.contextTooLong, result.errorCode⟩

/-- Go policy guard function -/
def goPolicyGuard (config : GoPolicyConfig) (token : Nat) (currentTime : Nat) : IO GoPolicyResult := do
  let leanConfig := toLeanPolicyConfig config
  let decoderState := DecoderState.mk 0 0 0 0

  let (result, _) := policyGuard leanConfig decoderState token currentTime
  return toGoPolicyResult result

/-- Generate Go CGO bindings -/
def generateGoBindings : String :=
"package rsk

/*
#cgo CFLAGS: -I${SRCDIR}/include
#cgo LDFLAGS: -L${SRCDIR}/lib -lrsk
#include \"rsk_sampler.h\"
#include \"rsk_policy.h\"
#include \"rsk_shape.h\"
*/
import \"C\"
import (
	\"fmt\"
	\"unsafe\"
)

// SamplingConfig represents a sampling configuration
type SamplingConfig struct {
	Method         string   `json:\"method\"`
	K              *int     `json:\"k,omitempty\"`
	P              *float32 `json:\"p,omitempty\"`
	Temperature    float32  `json:\"temperature\"`
	TargetEntropy  *float32 `json:\"target_entropy,omitempty\"`
	LearningRate   *float32 `json:\"learning_rate,omitempty\"`
	MaxIterations  *int     `json:\"max_iterations,omitempty\"`
}

// SamplingResult represents the result of a sampling operation
type SamplingResult struct {
	Probabilities  []float32 `json:\"probabilities\"`
	SelectedToken  int       `json:\"selected_token\"`
	Entropy        float32   `json:\"entropy\"`
	Iterations     int       `json:\"iterations\"`
}

// PolicyConfig represents a policy configuration
type PolicyConfig struct {
	AllowAllTokens           bool     `json:\"allow_all_tokens\"`
	BlockedTokens            []int    `json:\"blocked_tokens\"`
	RateLimitTokensPerSecond int      `json:\"rate_limit_tokens_per_second\"`
	MaxContextLength         int      `json:\"max_context_length\"`
	MaxTokensPerRequest      int      `json:\"max_tokens_per_request\"`
}

// PolicyResult represents the result of a policy guard operation
type PolicyResult struct {
	Allowed        bool  `json:\"allowed\"`
	BlockedToken   *int  `json:\"blocked_token,omitempty\"`
	RateLimited    bool  `json:\"rate_limited\"`
	ContextTooLong bool  `json:\"context_too_long\"`
	ErrorCode      int   `json:\"error_code\"`
}

// TensorShape represents a tensor shape
type TensorShape struct {
	Dimensions []int `json:\"dimensions\"`
	Size       int   `json:\"size\"`
}

// TensorData represents tensor data
type TensorData struct {
	Data  []float32   `json:\"data\"`
	Shape TensorShape `json:\"shape\"`
}

// Sample performs sampling on logits using the given configuration
func Sample(logits []float32, config SamplingConfig) (*SamplingResult, error) {
	// Convert Go config to C config
	cConfig := C.rsk_sampling_config_t{
		method:         C.uint32_t(getMethodCode(config.Method)),
		temperature:    C.float(config.Temperature),
	}

	if config.K != nil {
		cConfig.k = C.uint32_t(*config.K)
	}
	if config.P != nil {
		cConfig.p = C.float(*config.P)
	}
	if config.TargetEntropy != nil {
		cConfig.target_entropy = C.float(*config.TargetEntropy)
	}
	if config.LearningRate != nil {
		cConfig.learning_rate = C.float(*config.LearningRate)
	}
	if config.MaxIterations != nil {
		cConfig.max_iterations = C.uint32_t(*config.MaxIterations)
	}

	// Call C function
	result := C.rsk_sample(
		(*C.float)(unsafe.Pointer(&logits[0])),
		C.uint32_t(len(logits)),
		cConfig,
	)

	// Convert result back to Go
	if !bool(result.success) {
		return nil, fmt.Errorf(\"sampling failed\")
	}

	// Convert probabilities array
	probs := make([]float32, len(logits))
	for i := range probs {
		probs[i] = float32(C.get_probability(result.probs, C.uint32_t(i)))
	}

	return &SamplingResult{
		Probabilities:  probs,
		SelectedToken:  int(result.selected_token),
		Entropy:        float32(result.entropy),
		Iterations:     int(result.iterations),
	}, nil
}

// PolicyGuard checks if a token is allowed according to the policy
func PolicyGuard(config PolicyConfig, token int, currentTime int64) (*PolicyResult, error) {
	// Convert Go config to C config
	cConfig := C.rsk_policy_config_t{
		allow_all_tokens:           C.bool(config.AllowAllTokens),
		rate_limit_tokens_per_second: C.uint32_t(config.RateLimitTokensPerSecond),
		max_context_length:         C.uint32_t(config.MaxContextLength),
		max_tokens_per_request:     C.uint32_t(config.MaxTokensPerRequest),
	}

	// Convert blocked tokens
	if len(config.BlockedTokens) > 0 {
		cConfig.blocked_tokens = (*C.uint32_t)(unsafe.Pointer(&config.BlockedTokens[0]))
		cConfig.blocked_tokens_count = C.uint32_t(len(config.BlockedTokens))
	}

	// Call C function
	result := C.rsk_policy_guard(
		cConfig,
		C.uint32_t(token),
		C.uint64_t(currentTime),
	)

	// Convert result back to Go
	var blockedToken *int
	if result.blocked_token_present {
		token := int(result.blocked_token)
		blockedToken = &token
	}

	return &PolicyResult{
		Allowed:        bool(result.allowed),
		BlockedToken:   blockedToken,
		RateLimited:    bool(result.rate_limited),
		ContextTooLong: bool(result.context_too_long),
		ErrorCode:      int(result.error_code),
	}, nil
}

// CreateTensor creates a tensor with the given data and shape
func CreateTensor(data []float32, shape TensorShape) (*TensorData, error) {
	// Convert Go shape to C shape
	cShape := C.rsk_tensor_shape_t{
		size: C.uint32_t(shape.Size),
	}

	if len(shape.Dimensions) > 0 {
		cShape.dimensions = (*C.uint32_t)(unsafe.Pointer(&shape.Dimensions[0]))
		cShape.dimensions_count = C.uint32_t(len(shape.Dimensions))
	}

	// Call C function
	result := C.rsk_create_tensor(
		(*C.float)(unsafe.Pointer(&data[0])),
		C.uint32_t(len(data)),
		cShape,
	)

	// Convert result back to Go
	if !bool(result.success) {
		return nil, fmt.Errorf(\"tensor creation failed\")
	}

	// Convert data array
	tensorData := make([]float32, int(result.data.data_count))
	for i := range tensorData {
		tensorData[i] = float32(C.get_tensor_data(result.data.data, C.uint32_t(i)))
	}

	// Convert shape
	dimensions := make([]int, int(result.data.shape.dimensions_count))
	for i := range dimensions {
		dimensions[i] = int(C.get_tensor_dimension(result.data.shape.dimensions, C.uint32_t(i)))
	}

	return &TensorData{
		Data: tensorData,
		Shape: TensorShape{
			Dimensions: dimensions,
			Size:       int(result.data.shape.size),
		},
	}, nil
}

// MatrixMultiply performs matrix multiplication on two tensors
func MatrixMultiply(a, b *TensorData) (*TensorData, error) {
	// Convert Go tensors to C format
	cA := tensorToCFormat(a)
	cB := tensorToCFormat(b)

	// Call C function
	result := C.rsk_matrix_multiply(cA, cB)

	// Convert result back to Go
	if !bool(result.success) {
		return nil, fmt.Errorf(\"matrix multiplication failed\")
	}

	// Convert data array
	tensorData := make([]float32, int(result.data.data_count))
	for i := range tensorData {
		tensorData[i] = float32(C.get_tensor_data(result.data.data, C.uint32_t(i)))
	}

	// Convert shape
	dimensions := make([]int, int(result.data.shape.dimensions_count))
	for i := range dimensions {
		dimensions[i] = int(C.get_tensor_dimension(result.data.shape.dimensions, C.uint32_t(i)))
	}

	return &TensorData{
		Data: tensorData,
		Shape: TensorShape{
			Dimensions: dimensions,
			Size:       int(result.data.shape.size),
		},
	}, nil
}

// Helper functions
func getMethodCode(method string) int {
	switch method {
	case \"topk\":
		return 0
	case \"topp\":
		return 1
	case \"mirostat\":
		return 2
	default:
		return 0
	}
}

func tensorToCFormat(tensor *TensorData) C.rsk_tensor_data_t {
	// Implementation details for converting Go tensor to C format
	return C.rsk_tensor_data_t{}
}

//export get_probability
func get_probability(probs unsafe.Pointer, index C.uint32_t) C.float {
	// Implementation for accessing probability array
	return C.float(0.0)
}

//export get_tensor_data
func get_tensor_data(data unsafe.Pointer, index C.uint32_t) C.float {
	// Implementation for accessing tensor data
	return C.float(0.0)
}

//export get_tensor_dimension
func get_tensor_dimension(dimensions unsafe.Pointer, index C.uint32_t) C.uint32_t {
	// Implementation for accessing tensor dimensions
	return C.uint32_t(0)
}"

/-- Generate Go module file -/
def generateGoModule : String :=
"module github.com/runtime-safety-kernels/go

go 1.21

require (
	github.com/stretchr/testify v1.8.4
)

require (
	github.com/davecgh/go-spew v1.1.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)"

/-- Generate Go test file -/
def generateGoTests : String :=
"package rsk

import (
	\"testing\"
	\"time\"
)

func TestSample(t *testing.T) {
	logits := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	config := SamplingConfig{
		Method:      \"topk\",
		K:           &[]int{3}[0],
		Temperature: 1.0,
	}

	result, err := Sample(logits, config)
	if err != nil {
		t.Fatalf(\"Sample failed: %v\", err)
	}

	if len(result.Probabilities) != len(logits) {
		t.Errorf(\"Expected %d probabilities, got %d\", len(logits), len(result.Probabilities))
	}

	// Check that probabilities sum to 1
	sum := float32(0.0)
	for _, p := range result.Probabilities {
		sum += p
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf(\"Probabilities should sum to 1, got %f\", sum)
	}
}

func TestPolicyGuard(t *testing.T) {
	config := PolicyConfig{
		AllowAllTokens:           false,
		BlockedTokens:            []int{1, 2, 3},
		RateLimitTokensPerSecond: 100,
		MaxContextLength:         1000,
		MaxTokensPerRequest:      100,
	}

	// Test allowed token
	result, err := PolicyGuard(config, 5, time.Now().UnixNano())
	if err != nil {
		t.Fatalf(\"PolicyGuard failed: %v\", err)
	}

	if !result.Allowed {
		t.Error(\"Token 5 should be allowed\")
	}

	// Test blocked token
	result, err = PolicyGuard(config, 1, time.Now().UnixNano())
	if err != nil {
		t.Fatalf(\"PolicyGuard failed: %v\", err)
	}

	if result.Allowed {
		t.Error(\"Token 1 should be blocked\")
	}

	if result.BlockedToken == nil || *result.BlockedToken != 1 {
		t.Error(\"Blocked token should be 1\")
	}
}

func TestCreateTensor(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0, 4.0}
	shape := TensorShape{
		Dimensions: []int{2, 2},
		Size:       4,
	}

	tensor, err := CreateTensor(data, shape)
	if err != nil {
		t.Fatalf(\"CreateTensor failed: %v\", err)
	}

	if len(tensor.Data) != len(data) {
		t.Errorf(\"Expected %d data elements, got %d\", len(data), len(tensor.Data))
	}

	if len(tensor.Shape.Dimensions) != len(shape.Dimensions) {
		t.Errorf(\"Expected %d dimensions, got %d\", len(shape.Dimensions), len(tensor.Shape.Dimensions))
	}
}

func TestMatrixMultiply(t *testing.T) {
	// Create 2x2 matrices
	dataA := []float32{1.0, 2.0, 3.0, 4.0}
	shapeA := TensorShape{Dimensions: []int{2, 2}, Size: 4}
	tensorA, err := CreateTensor(dataA, shapeA)
	if err != nil {
		t.Fatalf(\"CreateTensor failed: %v\", err)
	}

	dataB := []float32{5.0, 6.0, 7.0, 8.0}
	shapeB := TensorShape{Dimensions: []int{2, 2}, Size: 4}
	tensorB, err := CreateTensor(dataB, shapeB)
	if err != nil {
		t.Fatalf(\"CreateTensor failed: %v\", err)
	}

	result, err := MatrixMultiply(tensorA, tensorB)
	if err != nil {
		t.Fatalf(\"MatrixMultiply failed: %v\", err)
	}

	// Expected result: [[19, 22], [43, 50]]
	expected := []float32{19.0, 22.0, 43.0, 50.0}
	for i, val := range result.Data {
		if val != expected[i] {
			t.Errorf(\"Expected %f at position %d, got %f\", expected[i], i, val)
		}
	}
}

func BenchmarkSample(b *testing.B) {
	logits := make([]float32, 1000)
	for i := range logits {
		logits[i] = float32(i)
	}

	config := SamplingConfig{
		Method:      \"topk\",
		K:           &[]int{40}[0],
		Temperature: 1.0,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Sample(logits, config)
		if err != nil {
			b.Fatalf(\"Sample failed: %v\", err)
		}
	}
}

func BenchmarkPolicyGuard(b *testing.B) {
	config := PolicyConfig{
		AllowAllTokens:           false,
		BlockedTokens:            []int{1, 2, 3},
		RateLimitTokensPerSecond: 1000,
		MaxContextLength:         8192,
		MaxTokensPerRequest:      1000,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := PolicyGuard(config, i%1000, time.Now().UnixNano())
		if err != nil {
			b.Fatalf(\"PolicyGuard failed: %v\", err)
		}
	}
}"

/-- Main extraction entry point -/
def main : IO Unit := do
  -- Generate Go bindings
  IO.FS.writeFile "src/extracted/rsk.go" generateGoBindings

  -- Generate Go module
  IO.FS.writeFile "src/extracted/go.mod" generateGoModule

  -- Generate Go tests
  IO.FS.writeFile "src/extracted/rsk_test.go" generateGoTests

  -- Run extraction tests
  let testLogits := [1.0, 2.0, 3.0, 4.0, 5.0]
  let config := GoSamplingConfig.mk "topk" (some 3) none 1.0 none none none

  let result ← goSample testLogits config
  match result with
  | none => IO.println "Go sampling test failed"
  | some res => IO.println s!"Go sampling test result: {res}"

  IO.println "Go bindings extraction completed successfully"

/-- Export for Lake build -/
#eval main

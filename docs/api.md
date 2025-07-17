# Runtime Safety Kernels API Reference

## Overview

The Runtime Safety Kernels (RSK) provides a comprehensive set of formally verified, high-performance components for AI model inference with guaranteed safety properties.

## Core Components

### RSK-1: Mathematically-Sound Samplers

#### Core Sampling Functions

```lean
-- Convert logits to probabilities using softmax
def logitsToProbs (logits : Logits n) : Probabilities n

-- Apply temperature scaling
def applyTemperature (logits : Logits n) (temp : Float) : Logits n

-- Unified sampling interface
def sample (logits : Logits n) (config : SamplingConfig) : SamplingResult n
```

#### Sampling Algorithms

**Top-K Sampling**

```lean
-- Create top-k configuration
def mkTopK (k : Nat) (temperature : Float := 1.0) : SamplingConfig

-- Top-k sampling with formal proofs
def topKSample (logits : Logits n) (config : TopKConfig) : TopKResult n
```

**Top-P (Nucleus) Sampling**

```lean
-- Create top-p configuration
def mkTopP (p : Float) (temperature : Float := 1.0) : SamplingConfig

-- Nucleus sampling with cumulative cutoff proofs
def topPSample (logits : Logits n) (config : TopPConfig) : TopPResult n
```

**Mirostat 2.0**

```lean
-- Create Mirostat configuration
def mkMirostat (targetEntropy : Float) (learningRate : Float := 0.1) (maxIterations : Nat := 100) : SamplingConfig

-- Mirostat 2.0 with entropy error bounds
def mirostatSample (logits : Logits n) (config : MirostatConfig) : MirostatResult n
```

#### Mathematical Properties

All sampling algorithms are formally proven to:

- Preserve probability simplex (sum to 1, non-negative)
- Maintain numerical stability
- Handle edge cases correctly
- Provide bounded entropy error (Mirostat)

### RSK-2: Race-Free Concurrent FSM

#### Concurrency Management

```lean
-- Initialize concurrency manager
def initConcurrencyManager (config : ConcurrencyConfig) : ConcurrencyManager

-- Submit request
def submitRequest (manager : ConcurrencyManager) (priority : Nat := 0) : ConcurrencyManager × RequestId

-- Start processing
def startProcessing (manager : ConcurrencyManager) (reqId : RequestId) (workerId : WorkerId) : Option ConcurrencyManager

-- Complete token
def completeToken (manager : ConcurrencyManager) (reqId : RequestId) (workerId : WorkerId) (tokenIdx : TokenIndex) : ConcurrencyManager
```

#### Safety Properties

The concurrency system is formally proven to be:

- **Deadlock-free**: No circular wait conditions
- **Fair**: Every pending request eventually gets processed
- **Race-free**: All state transitions are atomic
- **Bounded**: Queue length and latency are bounded

#### Performance Targets

- p99 queueing latency < 250 μs at 500 rps
- Support for ≤ 4,096 concurrent requests
- 64 worker threads with efficient scheduling

### RSK-3: Policy-Gated Decoding

#### Policy Management

```lean
-- Initialize policy manager
def initPolicyManager (config : PolicyConfig) (guard : PolicyGuard) : PolicyManager

-- Decode single token
def decodeToken (manager : PolicyManager) (token : Token) : IO (Except String Token)

-- Decode multiple tokens
def decodeTokens (manager : PolicyManager) (tokens : List Token) : IO (PolicyManager × List Token)
```

#### Built-in Policy Guards

```lean
-- Allow all tokens
def allowAllPolicy : PolicyGuard

-- Block specific tokens
def blockSpecificTokens (blockedTokens : List Token) : PolicyGuard

-- Rate limiting
def rateLimitPolicy (maxTokensPerSecond : Nat) : PolicyGuard

-- Context length checking
def contextLengthPolicy (maxContextLength : Nat) : PolicyGuard
```

#### Safety Guarantees

- **Complete Coverage**: Policy guard called on every token
- **Abort on Failure**: Decoder aborts immediately on guard failure
- **Performance**: < 10 μs per token policy check
- **Deterministic**: Same input always produces same result

### RSK-4: Shape-Safe Tensor API

#### Tensor Operations

```lean
-- Create tensor with shape validation
def mkTensorSafe (shape : Shape) (data : List Float) : Option (Tensor shape)

-- Safe tensor addition
def tensorAddSafe {shape : Shape} (t1 t2 : Tensor shape) : Option (Tensor shape)

-- Safe matrix multiplication
def tensorMatMulSafe {shape1 shape2 : Shape} (t1 : Tensor shape1) (t2 : Tensor shape2) : Option (Tensor (matMulShape shape1 shape2).getD [])

-- Safe tensor reshape
def tensorReshapeSafe {shape1 : Shape} (t : Tensor shape1) (shape2 : Shape) : Option (Tensor shape2)
```

#### Shape Validation

```lean
-- Validate shape at compile time
def validateShapeAtCompileTime (shape : Shape) : Bool

-- Check shape compatibility
def shapesCompatible (shape1 shape2 : Shape) : Bool
def shapesCompatibleForMatMul (shape1 shape2 : Shape) : Bool
def shapesCompatibleForBroadcast (shape1 shape2 : Shape) : Bool
```

#### Safety Properties

- **Compile-time Validation**: Shape mismatches caught at compile time
- **Runtime Safety**: All operations preserve tensor invariants
- **Memory Safety**: No buffer overflows or invalid access
- **Performance**: Minimal overhead for shape checking

## Language Bindings

### C/C++

```c
#include "tensor.h"

// Create tensor with compile-time shape validation
VALIDATE_SHAPE(shape, 2);
Tensor* tensor = create_tensor(shape, 2, data);

// Safe operations
Tensor* result = tensor_add(tensor1, tensor2);
Tensor* matmul = tensor_matmul(matrix1, matrix2);
```

### Python

```python
import rsk

# Sampling
logits = [1.0, 2.0, 3.0, 4.0, 5.0]
config = rsk.SamplingConfig.top_k(k=3, temperature=1.0)
result = rsk.sample(logits, config)

# Policy enforcement
policy = rsk.PolicyManager.allow_all()
token = policy.decode_token("test")

# Shape-safe tensors
tensor = rsk.Tensor([2, 3], data)
result = tensor.add(other_tensor)
```

### Rust

```rust
use rsk::{ConcurrencyManager, PolicyManager, Tensor};

// Concurrency
let mut manager = ConcurrencyManager::new(config);
let (new_manager, req_id) = manager.submit_request(priority);

// Policy enforcement
let mut policy = PolicyManager::new(config);
let result = policy.decode_token("test");

// Shape-safe tensors
let tensor = Tensor::new(shape, data)?;
let result = tensor.add(&other_tensor)?;
```

## Performance Benchmarks

### Sampling Performance

- **Top-K**: 4M tokens/s on Ryzen 9 single core
- **Top-P**: 3.5M tokens/s on Ryzen 9 single core
- **Mirostat**: 2M tokens/s on Ryzen 9 single core

### Concurrency Performance

- **Throughput**: 500 rps sustained
- **Latency**: p99 < 250 μs queueing latency
- **Scalability**: ≤ 4,096 concurrent requests

### Policy Enforcement

- **Latency**: < 10 μs per token
- **Throughput**: 100K tokens/s per core
- **Memory**: < 1MB per policy instance

### Shape-Safe Operations

- **Overhead**: < 5% compared to unsafe operations
- **Validation**: < 1 μs per shape check
- **Memory**: Zero-copy operations where possible

## Error Handling

All RSK components use structured error handling:

```lean
-- Sampling errors
inductive SamplingError
  | invalidConfig : String → SamplingError
  | numericalError : String → SamplingError

-- Policy errors
inductive PolicyError
  | blocked : String → PolicyError
  | rateLimited : Nat → PolicyError
  | contextExceeded : PolicyError

-- Shape errors
inductive ShapeError
  | invalidShape : Shape → ShapeError
  | incompatibleShapes : Shape → Shape → ShapeError
  | outOfBounds : Nat → ShapeError
```

## Configuration

### Sampling Configuration

```lean
structure SamplingConfig where
  algorithm : SamplingAlgorithm
  temperature : Float := 1.0
  maxTokens : Nat := 1000
  seed : Option Nat := none
```

### Concurrency Configuration

```lean
structure ConcurrencyConfig where
  maxWorkers : Nat := 64
  maxRequests : Nat := 4096
  queueTimeout : Nat := 1000
  workerTimeout : Nat := 5000
```

### Policy Configuration

```lean
structure PolicyConfig where
  maxTokens : Nat := 1000
  policyVersion : String := "1.0"
  enableRateLimiting : Bool := false
  maxTokensPerSecond : Nat := 100
  enableContextChecking : Bool := false
  maxContextLength : Nat := 1000
```

### Shape Configuration

```lean
structure ShapeConfig where
  enableCompileTimeChecks : Bool := true
  enableRuntimeChecks : Bool := true
  maxDimensions : Nat := 8
  maxShapeSize : Nat := 1000000
```

## Testing

### Unit Tests

```bash
lake exe tests
```

### Fuzzing

```bash
lake exe fuzz
```

### Benchmarks

```bash
lake exe benchmarks
```

### Integration Tests

```bash
lake exe tests --integration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement with formal proofs
4. Add comprehensive tests
5. Ensure CI passes
6. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

/--
Runtime Safety Kernels - Main Module

This is the main entry point for the Runtime Safety Kernels project,
providing access to all core components and utilities.
-/

import RuntimeSafetyKernels.Sampler
import RuntimeSafetyKernels.Concurrency
import RuntimeSafetyKernels.Policy
import RuntimeSafetyKernels.Shape
import RuntimeSafetyKernels.Fuzz
import RuntimeSafetyKernels.Benchmarks
import RuntimeSafetyKernels.LanguageBindings.Python
import RuntimeSafetyKernels.LanguageBindings.Go
import RuntimeSafetyKernels.LanguageBindings.NodeJS
import RuntimeSafetyKernels.LoadTesting
import RuntimeSafetyKernels.SIMD
import RuntimeSafetyKernels.PerformanceCI
import RuntimeSafetyKernels.BinaryOptimization
import RuntimeSafetyKernels.ThroughputOptimization

/-- Main module -/
module RuntimeSafetyKernels

/-- Version information -/
def version : String := "0.1.0"

/-- Project description -/
def description : String := "State-of-the-art runtime safety components for AI model inference"

/-- Main entry point for the application -/
def main (args : List String) : IO Unit := do
  IO.println s!"Runtime Safety Kernels v{version}"
  IO.println s!"{description}"
  IO.println "========================================"

  if args.isEmpty then
    IO.println "Available commands:"
    IO.println "  test        - Run all tests"
    IO.println "  benchmark   - Run performance benchmarks"
    IO.println "  fuzz        - Run fuzz testing"
    IO.println "  load-test   - Run load testing"
    IO.println "  simd        - Run SIMD optimization tests"
    IO.println "  perf-ci     - Run performance CI suite"
    IO.println "  extract     - Extract C/Rust kernels"
    IO.println "  python      - Generate Python bindings"
    IO.println "  go          - Generate Go bindings"
    IO.println "  nodejs      - Generate Node.js bindings"
    IO.println "  optimize-size - Optimize binary size (≤400KB)"
    IO.println "  optimize-throughput - Optimize throughput (≥4M tokens/s)"
    IO.println "  help        - Show this help message"
    return

  let command := args[0]

  match command with
  | "test" =>
    IO.println "Running all tests..."
    -- Run all test suites
    let _ ← RuntimeSafetyKernels.Tests.main []
    IO.println "All tests completed"

  | "benchmark" =>
    IO.println "Running performance benchmarks..."
    let _ ← RuntimeSafetyKernels.Benchmarks.main []
    IO.println "Benchmarks completed"

  | "fuzz" =>
    IO.println "Running fuzz testing..."
    let _ ← RuntimeSafetyKernels.Fuzz.main []
    IO.println "Fuzz testing completed"

  | "load-test" =>
    IO.println "Running load testing..."
    let _ ← RuntimeSafetyKernels.LoadTesting.main []
    IO.println "Load testing completed"

  | "simd" =>
    IO.println "Running SIMD optimization tests..."
    let _ ← RuntimeSafetyKernels.SIMD.main []
    IO.println "SIMD tests completed"

  | "perf-ci" =>
    IO.println "Running performance CI suite..."
    let _ ← RuntimeSafetyKernels.PerformanceCI.main (args.drop 1)
    IO.println "Performance CI completed"

  | "extract" =>
    IO.println "Extracting C/Rust kernels..."
    -- Run extraction for all components
    let _ ← RuntimeSafetyKernels.Sampler.Extract.main []
    let _ ← RuntimeSafetyKernels.Concurrency.Extract.main []
    let _ ← RuntimeSafetyKernels.Policy.Extract.main []
    let _ ← RuntimeSafetyKernels.Shape.Extract.main []
    IO.println "Kernel extraction completed"

  | "python" =>
    IO.println "Generating Python bindings..."
    let _ ← RuntimeSafetyKernels.LanguageBindings.Python.main []
    IO.println "Python bindings generated"

  | "go" =>
    IO.println "Generating Go bindings..."
    let _ ← RuntimeSafetyKernels.LanguageBindings.Go.main []
    IO.println "Go bindings generated"

  | "nodejs" =>
    IO.println "Generating Node.js bindings..."
    let _ ← RuntimeSafetyKernels.LanguageBindings.NodeJS.main []
    IO.println "Node.js bindings generated"

  | "optimize-size" =>
    IO.println "Running binary size optimization..."
    let _ ← RuntimeSafetyKernels.BinaryOptimization.main (args.drop 1)
    IO.println "Binary size optimization completed"

  | "optimize-throughput" =>
    IO.println "Running throughput optimization..."
    let _ ← RuntimeSafetyKernels.ThroughputOptimization.main (args.drop 1)
    IO.println "Throughput optimization completed"

  | "help" =>
    IO.println "Runtime Safety Kernels Help"
    IO.println "=========================="
    IO.println ""
    IO.println "This project provides state-of-the-art runtime safety components"
    IO.println "for AI model inference, including:"
    IO.println ""
    IO.println "Core Components:"
    IO.println "  • Sampling algorithms (top-k, nucleus, Mirostat 2.0)"
    IO.println "  • Concurrency state machine with deadlock freedom"
    IO.println "  • Policy-gated decoding with ultra-low latency"
    IO.println "  • Shape-safe tensor operations"
    IO.println ""
    IO.println "Performance Features:"
    IO.println "  • SIMD optimization (AVX2/AVX-512)"
    IO.println "  • Load testing for 4,096 concurrent requests"
    IO.println "  • Performance regression CI"
    IO.println "  • Comprehensive fuzz testing"
    IO.println ""
    IO.println "Language Bindings:"
    IO.println "  • Python (PyO3)"
    IO.println "  • Go (CGO)"
    IO.println "  • Node.js (N-API)"
    IO.println "  • C/Rust kernel extraction"
    IO.println ""
    IO.println "Optimization Features:"
    IO.println "  • Binary size optimization (≤400KB)"
    IO.println "  • Throughput optimization (≥4M tokens/s)"
    IO.println "  • Full AVX-512 SIMD support"
    IO.println ""
    IO.println "For more information, see the README.md file."

  | _ =>
    IO.println s!"Unknown command: {command}"
    IO.println "Run with 'help' to see available commands"
    IO.Process.exit 1

/-- Quick start example -/
def quickStartExample : IO Unit := do
  IO.println "Runtime Safety Kernels Quick Start Example"
  IO.println "=========================================="

  -- Example 1: Sampling
  IO.println "\n1. Sampling Example:"
  let logits := Vector.generate 1000 (fun i => Float.ofNat i)
  let config := SamplingConfig.topK ⟨40, 1.0⟩
  let result := sample logits config
  IO.println s!"   Top-K sampling result: {result}"

  -- Example 2: Policy Guarding
  IO.println "\n2. Policy Guarding Example:"
  let policyConfig := PolicyConfig.mk false [1, 2, 3] 1000 8192 1000
  let decoderState := DecoderState.mk 0 0 0 0
  let token := 100
  let currentTime := 1000
  let (policyResult, _) := policyGuard policyConfig decoderState token currentTime
  IO.println s!"   Policy guard result: {policyResult}"

  -- Example 3: Concurrency
  IO.println "\n3. Concurrency Example:"
  let request := RuntimeRequest.mk 1 1 100 1000 (Vector.mkArray 100 0)
  let state := ConcurrencyState.mk (Vector.mkArray 64 WorkerState.idle) [] 0 0 0
  let maybeNewState := submitRequest state request
  IO.println s!"   Request submitted: {maybeNewState.isSome}"

  -- Example 4: Tensor Operations
  IO.println "\n4. Tensor Operations Example:"
  let data := Vector.generate 100 (fun i => Float.ofNat i)
  let shape := TensorShape.mk #[10, 10]
  let tensorResult := createTensor data shape
  IO.println s!"   Tensor creation: {tensorResult}"

  -- Example 5: SIMD Optimization
  IO.println "\n5. SIMD Optimization Example:"
  let simdConfig := defaultSIMDConfig
  let simdProbs := simdLogitsToProbabilities logits simdConfig
  IO.println s!"   SIMD-optimized probabilities: {simdProbs.length} elements"

  IO.println "\nQuick start example completed successfully!"

/-- Export for Lake build -/
#eval main []

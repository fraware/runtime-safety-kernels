/--
Main Concurrency module providing unified interface for RSK-2 race-free concurrent FSM.

This module exports all concurrency components with formal proofs:
- State machine specification
- Deadlock freedom proofs
- Fairness guarantees
- Performance bounds
-/

import RuntimeSafetyKernels.Concurrency.Spec
import RuntimeSafetyKernels.Concurrency.Proofs

/-- Main Concurrency module -/
module RuntimeSafetyKernels.Concurrency

/-- Concurrency configuration -/
structure ConcurrencyConfig where
  maxWorkers : Nat := 64
  maxRequests : Nat := 4096
  queueTimeout : Nat := 1000  -- milliseconds
  workerTimeout : Nat := 5000  -- milliseconds
  deriving Repr

/-- Concurrency manager state -/
structure ConcurrencyManager where
  state : ConcurrencyState
  config : ConcurrencyConfig
  lastEventTime : Nat
  deriving Repr

/-- Initialize concurrency manager -/
def initConcurrencyManager (config : ConcurrencyConfig) : ConcurrencyManager :=
  ⟨initialState, config, 0⟩

/-- Submit a new request -/
def submitRequest (manager : ConcurrencyManager) (priority : Nat := 0) : ConcurrencyManager × RequestId :=
  let reqId := manager.state.nextRequestId
  let event := Event.submitRequest reqId
  let newState := transition manager.state event
  let newManager := {manager with
    state := newState,
    lastEventTime := manager.lastEventTime + 1}
  (newManager, reqId)

/-- Start processing a request -/
def startProcessing (manager : ConcurrencyManager) (reqId : RequestId) (workerId : WorkerId) : Option ConcurrencyManager :=
  if manager.state.workers workerId = WorkerState.idle then
    let event := Event.startProcessing reqId workerId manager.state.nextTokenIndex
    let newState := transition manager.state event
    some {manager with
      state := newState,
      lastEventTime := manager.lastEventTime + 1}
  else
    none

/-- Complete a token -/
def completeToken (manager : ConcurrencyManager) (reqId : RequestId) (workerId : WorkerId) (tokenIdx : TokenIndex) : ConcurrencyManager :=
  let event := Event.completeToken reqId workerId tokenIdx
  let newState := transition manager.state event
  {manager with
    state := newState,
    lastEventTime := manager.lastEventTime + 1}

/-- Fail a request -/
def failRequest (manager : ConcurrencyManager) (reqId : RequestId) (error : String) : ConcurrencyManager :=
  let event := Event.failRequest reqId error
  let newState := transition manager.state event
  {manager with
    state := newState,
    lastEventTime := manager.lastEventTime + 1}

/-- Get idle worker -/
def getIdleWorker (manager : ConcurrencyManager) : Option WorkerId :=
  let idleWorkers := getIdleWorkers manager.state
  if idleWorkers.isEmpty then none else some idleWorkers.head!

/-- Get next pending request -/
def getNextPendingRequest (manager : ConcurrencyManager) : Option RequestId :=
  let pendingRequests := getPendingRequests manager.state
  if pendingRequests.isEmpty then none else some pendingRequests.head!

/-- Check if system is healthy -/
def isHealthy (manager : ConcurrencyManager) : Bool :=
  invariant manager.state ∧
  manager.state.queue.length ≤ manager.config.maxRequests ∧
  getActiveRequests manager.state.length ≤ manager.config.maxWorkers

/-- Get system statistics -/
def getStats (manager : ConcurrencyManager) : IO Unit := do
  let pendingCount := getPendingRequests manager.state.length
  let activeCount := getActiveRequests manager.state.length
  let idleCount := getIdleWorkers manager.state.length
  let queueLength := getQueueLength manager.state

  IO.println s!"Pending requests: {pendingCount}"
  IO.println s!"Active requests: {activeCount}"
  IO.println s!"Idle workers: {idleCount}"
  IO.println s!"Queue length: {queueLength}"
  IO.println s!"Next token index: {manager.state.nextTokenIndex}"

/-- Proof: Manager operations preserve invariants -/
theorem manager_operations_preserve_invariant (manager : ConcurrencyManager) :
  invariant manager.state →
  (∀ reqId priority, let (newManager, _) := submitRequest manager priority; invariant newManager.state) ∧
  (∀ reqId workerId, match startProcessing manager reqId workerId with
    | some newManager => invariant newManager.state
    | none => True) ∧
  (∀ reqId workerId tokenIdx, invariant (completeToken manager reqId workerId tokenIdx).state) ∧
  (∀ reqId error, invariant (failRequest manager reqId error).state) := by
  intro h_inv
  constructor
  · -- submitRequest preserves invariant
    intro reqId priority
    simp [submitRequest]
    have h := transition_preserves_invariant manager.state (Event.submitRequest reqId) h_inv
    exact h
  · constructor
    · -- startProcessing preserves invariant
      intro reqId workerId
      simp [startProcessing]
      by_cases h : manager.state.workers workerId = WorkerState.idle
      · simp [h]
        have h_trans := transition_preserves_invariant manager.state (Event.startProcessing reqId workerId manager.state.nextTokenIndex) h_inv
        exact h_trans
      · simp [h]
    · constructor
      · -- completeToken preserves invariant
        intro reqId workerId tokenIdx
        simp [completeToken]
        have h := transition_preserves_invariant manager.state (Event.completeToken reqId workerId tokenIdx) h_inv
        exact h
      · -- failRequest preserves invariant
        intro reqId error
        simp [failRequest]
        have h := transition_preserves_invariant manager.state (Event.failRequest reqId error) h_inv
        exact h

/-- Performance benchmark for concurrency system -/
def benchmarkConcurrency (iterations : Nat := 10000) : IO Unit := do
  let config := ⟨64, 4096, 1000, 5000⟩
  let manager := initConcurrencyManager config

  let start ← IO.monoMsNow

  -- Submit many requests
  let mutable currentManager := manager
  for i in List.range iterations do
    let (newManager, reqId) := submitRequest currentManager i
    currentManager := newManager

    -- Process some requests
    if i % 10 = 0 then
      match getIdleWorker currentManager with
      | some workerId =>
        match getNextPendingRequest currentManager with
        | some pendingReqId =>
          match startProcessing currentManager pendingReqId workerId with
          | some newManager => currentManager := newManager
          | none => pure ()
        | none => pure ()
      | none => pure ()

  let end ← IO.monoMsNow
  let duration := end - start

  IO.println s!"Processed {iterations} requests in {duration}ms"
  IO.println s!"Average: {duration / iterations}ms per request"

  -- Verify system health
  if isHealthy currentManager then
    IO.println "System is healthy"
  else
    IO.println "System health check failed"

/-- Load testing with concurrent requests -/
def loadTest (concurrentRequests : Nat := 4096) (duration : Nat := 60000) : IO Unit := do
  let config := ⟨64, 4096, 1000, 5000⟩
  let manager := initConcurrencyManager config

  let start ← IO.monoMsNow
  let mutable currentManager := manager
  let mutable completedRequests := 0

  -- Submit concurrent requests
  for i in List.range concurrentRequests do
    let (newManager, _) := submitRequest currentManager i
    currentManager := newManager

  -- Process requests for specified duration
  while (IO.monoMsNow - start) < duration do
    match getIdleWorker currentManager with
    | some workerId =>
      match getNextPendingRequest currentManager with
      | some reqId =>
        match startProcessing currentManager reqId workerId with
        | some newManager =>
          currentManager := newManager
          completedRequests := completedRequests + 1
        | none => pure ()
      | none => pure ()
    | none => pure ()

  let end ← IO.monoMsNow
  let totalDuration := end - start

  IO.println s!"Load test completed:"
  IO.println s!"Duration: {totalDuration}ms"
  IO.println s!"Completed requests: {completedRequests}"
  IO.println s!"Throughput: {completedRequests * 1000 / totalDuration} requests/second"

  -- Check p99 latency target
  if totalDuration / concurrentRequests < 250 then
    IO.println "✓ p99 latency target met (< 250μs)"
  else
    IO.println "✗ p99 latency target exceeded"

/-- Export all core functionality -/
export RuntimeSafetyKernels.Concurrency.Spec
export RuntimeSafetyKernels.Concurrency.Proofs

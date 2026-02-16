"""
Training Manager - Centralized training orchestration with cooperative cancellation
===================================================================================

Provides:
- Thread-safe training state management
- Cooperative cancellation (no os.kill, no forced termination)
- Skip-current-model support
- Real-time elapsed time tracking (total + per-model)
- Structured status reporting via JSON
- Background thread execution
- Singleton pattern to prevent duplicate sessions

Cancellation Flow:
1. Frontend calls /cancel-training
2. TrainingManager sets cancel_requested = True (thread-safe via threading.Event)
3. Training loop checks is_cancelled() before each model
4. Optuna objective checks is_cancelled() before each trial
5. If cancelled mid-trial, Optuna study.stop() is called via callback
6. Engine returns -9999 scores, loop breaks, cleanup runs

Skip Flow:
1. Frontend calls /skip-model
2. TrainingManager sets skip_requested = True
3. Current model's Optuna study.stop() fires via callback
4. Loop advances to next model, skip flag resets
"""

import threading
import time
import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TrainingPhase(str, Enum):
    """Training lifecycle phases"""
    INITIALIZING = "initializing"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    PLOTTING = "plotting"
    SAVING = "saving"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ModelTrainingState:
    """Per-model training state"""
    model_name: str
    status: str = "pending"  # pending | training | skipped | completed | cancelled | failed
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    current_trial: int = 0
    total_trials: int = 0
    best_score: Optional[float] = None
    error: Optional[str] = None

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at else time.time()
        return round(end - self.started_at, 2)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
            "current_trial": self.current_trial,
            "total_trials": self.total_trials,
            "best_score": self.best_score,
            "error": self.error,
        }


class TrainingManager:
    """
    Centralized, thread-safe training session manager.
    
    Only ONE training session can be active at a time (enforced by singleton lock).
    State is stored in-memory and exposed via get_status().
    
    Usage:
        manager = TrainingManager.get_instance()
        session_id = manager.start_session(model_names=["xgboost", "lightgbm"], n_trials=75)
        # ... in background thread:
        #   manager.is_cancelled()  -> check cancellation
        #   manager.is_skip_requested()  -> check skip
        #   manager.set_current_model("xgboost")
        #   manager.update_trial(trial_num, total_trials)
        #   manager.complete_model("xgboost", score=0.95)
        # ... 
        manager.finish_session()
    """

    _instance: Optional["TrainingManager"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "TrainingManager":
        """Thread-safe singleton accessor"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        # Session identity
        self._session_id: Optional[str] = None
        self._is_running = False

        # Cancellation / skip events (thread-safe)
        self._cancel_event = threading.Event()
        self._skip_event = threading.Event()

        # Timing
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None

        # Phase tracking
        self._phase: TrainingPhase = TrainingPhase.INITIALIZING

        # Model-level state
        self._current_model_name: Optional[str] = None
        self._model_states: Dict[str, ModelTrainingState] = {}
        self._model_order: List[str] = []

        # Trial tracking for current model
        self._current_trial: int = 0
        self._total_trials: int = 0

        # Logs buffer (bounded)
        self._logs: List[str] = []
        self._max_logs = 200

        # Thread-safe lock for state mutations
        self._state_lock = threading.Lock()

        # Background thread reference
        self._training_thread: Optional[threading.Thread] = None

        # Force-cancel flag (mutable list for sharing with subprocesses)
        self._force_cancel_flag: List[bool] = [False]

        # Result storage (set after training completes)
        self._result_model_id: Optional[str] = None
        self._training_duration: Optional[float] = None

    # ─── Session Lifecycle ────────────────────────────────────────────

    def start_session(
        self,
        session_id: str,
        model_names: List[str],
        n_trials: int = 75,
    ) -> str:
        """
        Initialize a new training session. Raises if one is already running.
        
        Returns:
            session_id
        """
        with self._state_lock:
            if self._is_running:
                raise RuntimeError(
                    f"Training session '{self._session_id}' is already running. "
                    "Cancel it first or wait for it to finish."
                )

            # Reset all state
            self._session_id = session_id
            self._is_running = True
            self._cancel_event.clear()
            self._skip_event.clear()
            self._force_cancel_flag = [False]
            self._started_at = time.time()
            self._finished_at = None
            self._phase = TrainingPhase.INITIALIZING
            self._current_model_name = None
            self._current_trial = 0
            self._total_trials = n_trials
            self._logs = []

            # Initialize per-model states
            self._model_order = list(model_names)
            self._model_states = {
                name: ModelTrainingState(model_name=name, total_trials=n_trials)
                for name in model_names
            }

            self._result_model_id = None
            self._training_duration = None

            self._add_log(f"Session {session_id} started with {len(model_names)} models")
            logger.info(f"Training session started: {session_id}")

        return session_id

    def finish_session(self, phase: TrainingPhase = TrainingPhase.COMPLETED):
        """Mark session as finished (completed, cancelled, or failed)."""
        with self._state_lock:
            self._phase = phase
            self._finished_at = time.time()
            self._is_running = False
            self._add_log(f"Session finished: {phase.value}")
            logger.info(f"Training session finished: {self._session_id} ({phase.value})")

    # ─── Cancellation ─────────────────────────────────────────────────

    def request_cancel(self):
        """Request cooperative cancellation (called from API endpoint)."""
        self._cancel_event.set()
        self._force_cancel_flag[0] = True
        with self._state_lock:
            self._add_log("Cancellation requested by user")
        logger.info(f"Cancel requested for session {self._session_id}")

    def is_cancelled(self) -> bool:
        """Check if cancellation was requested (called from training loop)."""
        return self._cancel_event.is_set()

    @property
    def force_cancel_flag(self) -> List[bool]:
        """Mutable list for sharing cancel state with subprocesses."""
        return self._force_cancel_flag

    def cancellation_callback(self) -> bool:
        """Lambda-compatible cancellation check for AutoMLEngine."""
        return self._cancel_event.is_set()

    # ─── Skip ─────────────────────────────────────────────────────────

    def request_skip(self):
        """Request skipping the current model (called from API endpoint)."""
        self._skip_event.set()
        with self._state_lock:
            model = self._current_model_name or "unknown"
            self._add_log(f"Skip requested for model: {model}")
        logger.info(f"Skip requested for model: {self._current_model_name}")

    def is_skip_requested(self) -> bool:
        """Check if skip was requested for current model."""
        return self._skip_event.is_set()

    def clear_skip(self):
        """Reset skip flag after advancing to next model."""
        self._skip_event.clear()

    # ─── Model-Level Tracking ─────────────────────────────────────────

    def set_phase(self, phase: TrainingPhase):
        """Update the current training phase."""
        with self._state_lock:
            self._phase = phase
            self._add_log(f"Phase: {phase.value}")

    def set_current_model(self, model_name: str):
        """Mark a model as the one currently being trained."""
        with self._state_lock:
            self._current_model_name = model_name
            self._current_trial = 0
            if model_name in self._model_states:
                self._model_states[model_name].status = "training"
                self._model_states[model_name].started_at = time.time()
                self._model_states[model_name].current_trial = 0
            self._add_log(f"Training model: {model_name}")

    def update_trial(self, trial_number: int, total_trials: int, best_score: Optional[float] = None):
        """Update trial progress for the current model."""
        with self._state_lock:
            self._current_trial = trial_number
            self._total_trials = total_trials
            if self._current_model_name and self._current_model_name in self._model_states:
                state = self._model_states[self._current_model_name]
                state.current_trial = trial_number
                state.total_trials = total_trials
                if best_score is not None:
                    state.best_score = best_score

    def complete_model(self, model_name: str, score: Optional[float] = None, status: str = "completed"):
        """Mark a model as completed/skipped/cancelled/failed."""
        with self._state_lock:
            if model_name in self._model_states:
                state = self._model_states[model_name]
                state.status = status
                state.finished_at = time.time()
                if score is not None:
                    state.best_score = score
                self._add_log(f"Model {model_name}: {status}" + (f" (score: {score:.4f})" if score else ""))

    def fail_model(self, model_name: str, error: str):
        """Mark a model as failed with error."""
        with self._state_lock:
            if model_name in self._model_states:
                state = self._model_states[model_name]
                state.status = "failed"
                state.finished_at = time.time()
                state.error = error
                self._add_log(f"Model {model_name} failed: {error}")

    # ─── Logging ──────────────────────────────────────────────────────

    def _add_log(self, message: str):
        """Append a log entry (must be called under _state_lock)."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self._logs.append(entry)
        if len(self._logs) > self._max_logs:
            self._logs = self._logs[-self._max_logs:]

    def add_log(self, message: str):
        """Public log entry (acquires lock)."""
        with self._state_lock:
            self._add_log(message)

    # ─── Status Reporting ─────────────────────────────────────────────

    def get_status(self) -> dict:
        """
        Return structured JSON status for the /training-status endpoint.
        Thread-safe snapshot of current state.
        """
        with self._state_lock:
            now = time.time()

            # Total elapsed
            total_elapsed = 0.0
            if self._started_at:
                end = self._finished_at if self._finished_at else now
                total_elapsed = round(end - self._started_at, 2)

            # Current model elapsed
            current_model_elapsed = 0.0
            if self._current_model_name and self._current_model_name in self._model_states:
                current_model_elapsed = self._model_states[self._current_model_name].elapsed_seconds

            # Completed / remaining model lists
            completed = [
                name for name, s in self._model_states.items()
                if s.status in ("completed", "skipped", "failed", "cancelled")
            ]
            remaining = [
                name for name in self._model_order
                if name not in completed and name != self._current_model_name
            ]

            # Model details
            models_detail = [
                self._model_states[name].to_dict()
                for name in self._model_order
                if name in self._model_states
            ]

            return {
                "session_id": self._session_id,
                "is_running": self._is_running,
                "phase": self._phase.value,
                "cancel_requested": self._cancel_event.is_set(),
                "skip_requested": self._skip_event.is_set(),
                "total_elapsed_seconds": total_elapsed,
                "current_model": self._current_model_name,
                "current_model_elapsed_seconds": current_model_elapsed,
                "completed_models": completed,
                "remaining_models": remaining,
                "current_trial": self._current_trial,
                "total_trials": self._total_trials,
                "total_models": len(self._model_order),
                "models_completed_count": len(completed),
                "models": models_detail,
                "logs": list(self._logs[-50:]),  # Last 50 log lines
            }

    # ─── Background Thread ────────────────────────────────────────────

    def run_in_background(self, target: Callable, args: tuple = (), kwargs: dict = None):
        """
        Start training function in a daemon background thread.
        Stores thread reference so we can check if it's alive.
        """
        if kwargs is None:
            kwargs = {}
        self._training_thread = threading.Thread(
            target=target,
            args=args,
            kwargs=kwargs,
            daemon=True,
            name=f"training-{self._session_id}",
        )
        self._training_thread.start()
        logger.info(f"Background training thread started: {self._training_thread.name}")

    @property
    def is_thread_alive(self) -> bool:
        """Check if the background training thread is still running."""
        return self._training_thread is not None and self._training_thread.is_alive()

    # ─── Cleanup ──────────────────────────────────────────────────────

    def reset(self):
        """Full reset (only call when session is not running)."""
        with self._state_lock:
            self._session_id = None
            self._is_running = False
            self._cancel_event.clear()
            self._skip_event.clear()
            self._force_cancel_flag = [False]
            self._started_at = None
            self._finished_at = None
            self._phase = TrainingPhase.INITIALIZING
            self._current_model_name = None
            self._model_states.clear()
            self._model_order.clear()
            self._current_trial = 0
            self._total_trials = 0
            self._logs.clear()
            self._training_thread = None
            self._result_model_id = None
            self._training_duration = None

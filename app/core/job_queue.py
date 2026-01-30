"""
Multi-worker file-based job queue system for background processing.

Design notes:
- Uses filesystem for persistence (no external dependencies)
- Supports multiple concurrent workers via ThreadPoolExecutor
- Supports progress tracking and cancellation
- Thread-safe for multi-worker scenarios
"""

import os
import json
import uuid
import time
import threading
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Dict, Any, List, Set
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
import signal


# Global process tracker for cancellation support
class ProcessTracker:
    """Track running subprocess PIDs by job ID for cancellation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._job_pids: Dict[str, Set[int]] = {}

    def register(self, job_id: str, pid: int):
        """Register a subprocess PID for a job."""
        with self._lock:
            if job_id not in self._job_pids:
                self._job_pids[job_id] = set()
            self._job_pids[job_id].add(pid)

    def unregister(self, job_id: str, pid: int):
        """Unregister a subprocess PID."""
        with self._lock:
            if job_id in self._job_pids:
                self._job_pids[job_id].discard(pid)

    def kill_all(self, job_id: str) -> int:
        """Kill all processes for a job. Returns count of killed processes."""
        with self._lock:
            pids = self._job_pids.pop(job_id, set()).copy()

        killed = 0
        for pid in pids:
            try:
                # Kill the process and its children using process group
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                killed += 1
            except (ProcessLookupError, PermissionError, OSError):
                # Try killing just the process
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                except:
                    pass
        return killed

    def cleanup(self, job_id: str):
        """Remove all tracking for a job."""
        with self._lock:
            self._job_pids.pop(job_id, None)


# Global instance
_process_tracker = ProcessTracker()


def get_process_tracker() -> ProcessTracker:
    """Get the global process tracker."""
    return _process_tracker


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Progress information for a running job."""
    current: int = 0
    total: int = 0
    message: str = ""
    sub_progress: float = 0.0  # 0-1 for current item
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    worker_id: Optional[int] = None

    @property
    def percent(self) -> float:
        if self.total == 0:
            return 0.0
        base = (self.current / self.total) * 100
        sub = (self.sub_progress / self.total) * 100 if self.total > 0 else 0
        return min(base + sub, 100.0)


@dataclass
class Job:
    """Represents a background job."""
    id: str
    job_type: str  # e.g., "slbl_batch", "single_simulation", "simulation_sweep"
    project_name: str
    created_at: str
    created_by: str
    status: JobStatus = JobStatus.PENDING
    params: Dict[str, Any] = field(default_factory=dict)
    progress: JobProgress = field(default_factory=JobProgress)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'Job':
        data['status'] = JobStatus(data['status'])
        data['progress'] = JobProgress(**data.get('progress', {}))
        return cls(**data)


class JobQueue:
    """
    File-based job queue with multiple background workers.

    Jobs are stored as JSON files in the queue directory.
    Multiple worker threads process jobs concurrently using ThreadPoolExecutor.
    """

    def __init__(self, queue_dir: Path, max_workers: int = 4):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self._handlers: Dict[str, Callable] = {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running_jobs: Set[str] = set()  # Track job IDs currently being processed
        self._futures: Dict[str, Future] = {}  # Track futures by job ID

    def register_handler(self, job_type: str, handler: Callable):
        """
        Register a handler function for a job type.

        Handler signature: handler(job: Job, progress_callback: Callable) -> dict
        Progress callback: progress_callback(current, total, message, sub_progress)
        """
        self._handlers[job_type] = handler

    def submit(self, job_type: str, project_name: str, created_by: str,
               params: dict) -> Job:
        """Submit a new job to the queue."""
        job = Job(
            id=str(uuid.uuid4())[:8],
            job_type=job_type,
            project_name=project_name,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            params=params
        )
        self._save_job(job)
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        job_path = self.queue_dir / f"{job_id}.json"
        if not job_path.exists():
            return None

        try:
            with self._lock:
                with open(job_path) as f:
                    return Job.from_dict(json.load(f))
        except Exception:
            return None

    def list_jobs(self, project_name: Optional[str] = None,
                  status: Optional[JobStatus] = None,
                  limit: int = 50) -> List[Job]:
        """List jobs, optionally filtered by project and status."""
        jobs = []

        for job_file in sorted(self.queue_dir.glob("*.json"),
                               key=lambda p: p.stat().st_mtime, reverse=True):
            if len(jobs) >= limit:
                break

            try:
                with open(job_file) as f:
                    job = Job.from_dict(json.load(f))

                if project_name and job.project_name != project_name:
                    continue
                if status and job.status != status:
                    continue

                jobs.append(job)
            except Exception:
                continue

        return jobs

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job and kill any running processes."""
        job = self.get_job(job_id)
        if job and job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            # Kill any running processes for this job
            killed = _process_tracker.kill_all(job_id)
            if killed > 0:
                print(f"[JobQueue] Killed {killed} processes for job {job_id}")

            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now().isoformat()
            self._save_job(job)
            return True
        return False

    def delete_job(self, job_id: str, delete_files: bool = True) -> bool:
        """Delete a job file and optionally its output files.

        Args:
            job_id: The job ID to delete
            delete_files: If True, also delete output directories on disk

        Returns:
            True if job was deleted, False otherwise
        """
        import shutil

        job = self.get_job(job_id)
        if job and job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            # Delete output files if requested
            if delete_files:
                output_dirs = []

                # Get output directory from result
                if job.result:
                    result_dir = job.result.get('output_dir') or job.result.get('sim_dir')
                    if result_dir:
                        output_dirs.append(result_dir)

                # For probability_ensemble cancelled/failed jobs, search project probability dir
                if job.job_type == 'probability_ensemble' and not output_dirs:
                    try:
                        from core.project_manager import Project
                        project = Project.load(job.project_name)
                        if project.probability_dir.exists():
                            # Find ensemble directories created around job start time
                            job_created = datetime.fromisoformat(job.created_at)
                            for ensemble_dir in project.probability_dir.iterdir():
                                if ensemble_dir.is_dir() and ensemble_dir.name.startswith('ensemble_'):
                                    # Check if directory was created within 1 minute of job
                                    dir_mtime = datetime.fromtimestamp(ensemble_dir.stat().st_mtime)
                                    time_diff = abs((dir_mtime - job_created).total_seconds())
                                    if time_diff < 60:  # Within 1 minute
                                        output_dirs.append(str(ensemble_dir))
                    except Exception as e:
                        print(f"[JobQueue] Warning: Could not search for ensemble dirs: {e}")

                # Delete all found output directories
                for output_dir in output_dirs:
                    output_path = Path(output_dir)
                    if output_path.exists() and output_path.is_dir():
                        try:
                            shutil.rmtree(output_path)
                            print(f"[JobQueue] Deleted output directory: {output_path}")
                        except Exception as e:
                            print(f"[JobQueue] Warning: Could not delete {output_path}: {e}")

            # Delete the job file
            job_path = self.queue_dir / f"{job_id}.json"
            if job_path.exists():
                with self._lock:
                    job_path.unlink()
                return True
        return False

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status including worker info."""
        all_jobs = self.list_jobs(limit=100)
        running_jobs = [j for j in all_jobs if j.status == JobStatus.RUNNING]

        # Count active workers from running jobs (each running job uses a worker)
        # Also track unique worker IDs for jobs that have them
        active_worker_ids = set()
        for job in running_jobs:
            if job.progress.worker_id is not None:
                active_worker_ids.add(job.progress.worker_id)

        # Active workers = number of running jobs (each job gets one worker)
        active_workers = len(running_jobs)

        return {
            'max_workers': self.max_workers,
            'active_workers': active_workers,
            'running_jobs': len(running_jobs),
            'pending_jobs': len([j for j in all_jobs if j.status == JobStatus.PENDING]),
            'failed_jobs': len([j for j in all_jobs if j.status == JobStatus.FAILED]),
        }

    def _save_job(self, job: Job):
        """Save job to filesystem."""
        job_path = self.queue_dir / f"{job.id}.json"
        with self._lock:
            with open(job_path, 'w') as f:
                json.dump(job.to_dict(), f, indent=2)

    def _get_pending_jobs(self, limit: int = 10) -> List[Job]:
        """Get pending jobs sorted by creation time (FIFO)."""
        pending = []
        for job_file in sorted(self.queue_dir.glob("*.json"),
                               key=lambda p: p.stat().st_mtime):
            if len(pending) >= limit:
                break
            try:
                with open(job_file) as f:
                    job = Job.from_dict(json.load(f))
                if job.status == JobStatus.PENDING and job.id not in self._running_jobs:
                    pending.append(job)
            except Exception:
                continue
        return pending

    def _process_job(self, job: Job, worker_id: int):
        """Process a single job (runs in worker thread)."""
        handler = self._handlers.get(job.job_type)
        if not handler:
            job.status = JobStatus.FAILED
            job.error = f"No handler registered for job type: {job.job_type}"
            self._save_job(job)
            return

        # Update status to running
        job.status = JobStatus.RUNNING
        job.progress.started_at = datetime.now().isoformat()
        job.progress.worker_id = worker_id
        self._save_job(job)

        print(f"[Worker-{worker_id}] Starting job {job.id}: {job.job_type}")

        # Progress callback - checks for cancellation
        def progress_callback(current: int, total: int, message: str = "",
                              sub_progress: float = 0.0):
            # Check if job was cancelled
            current_job = self.get_job(job.id)
            if current_job and current_job.status == JobStatus.CANCELLED:
                raise InterruptedError("Job was cancelled")

            job.progress.current = current
            job.progress.total = total
            job.progress.message = message
            job.progress.sub_progress = sub_progress
            job.progress.updated_at = datetime.now().isoformat()
            self._save_job(job)

        try:
            result = handler(job, progress_callback)

            # Check if cancelled during processing
            current_job = self.get_job(job.id)
            if current_job and current_job.status == JobStatus.CANCELLED:
                print(f"[Worker-{worker_id}] Job {job.id} was cancelled")
                return

            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = datetime.now().isoformat()
            print(f"[Worker-{worker_id}] Completed job {job.id}")

        except InterruptedError:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now().isoformat()
            print(f"[Worker-{worker_id}] Job {job.id} cancelled")

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            job.completed_at = datetime.now().isoformat()
            print(f"[Worker-{worker_id}] Job {job.id} failed: {e}")

        self._save_job(job)

    def _job_done_callback(self, job_id: str, future: Future):
        """Called when a job future completes."""
        with self._lock:
            self._running_jobs.discard(job_id)
            self._futures.pop(job_id, None)

    def _dispatcher_loop(self):
        """Main dispatcher loop - assigns pending jobs to available workers."""
        worker_counter = 0

        while not self._stop_event.is_set():
            # Clean up completed futures
            with self._lock:
                completed = [jid for jid, f in self._futures.items() if f.done()]
                for jid in completed:
                    self._running_jobs.discard(jid)
                    self._futures.pop(jid, None)

            # Check available worker slots
            available_slots = self.max_workers - len(self._running_jobs)

            if available_slots > 0:
                # Get pending jobs
                pending_jobs = self._get_pending_jobs(limit=available_slots)

                for job in pending_jobs:
                    if job.id in self._running_jobs:
                        continue

                    # Mark as running to prevent duplicate dispatch
                    with self._lock:
                        if job.id in self._running_jobs:
                            continue
                        self._running_jobs.add(job.id)

                    # Assign worker ID
                    worker_counter += 1
                    worker_id = worker_counter % 1000

                    # Submit to thread pool
                    future = self._executor.submit(self._process_job, job, worker_id)
                    future.add_done_callback(
                        lambda f, jid=job.id: self._job_done_callback(jid, f)
                    )

                    with self._lock:
                        self._futures[job.id] = future

            # Wait before checking again
            time.sleep(1)

    def start_worker(self):
        """Start the background worker pool."""
        if self._executor is not None:
            return

        self._stop_event.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="JobWorker"
        )

        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            daemon=True,
            name="JobDispatcher"
        )
        self._dispatcher_thread.start()

        print(f"[JobQueue] Started with {self.max_workers} workers")

    def stop_worker(self):
        """Stop the background worker pool."""
        self._stop_event.set()

        if self._dispatcher_thread:
            self._dispatcher_thread.join(timeout=5)

        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

        print("[JobQueue] Workers stopped")


# Global job queue instance
_job_queue: Optional[JobQueue] = None


def _register_all_handlers(job_queue: JobQueue):
    """Register all job handlers. Safe to call multiple times."""
    try:
        from core.slbl_handler import register_slbl_handlers
        from core.simulation_handler import register_simulation_handlers
        register_slbl_handlers(job_queue)
        register_simulation_handlers(job_queue)
        print(f"[JobQueue] Handlers registered: {list(job_queue._handlers.keys())}")
    except Exception as e:
        print(f"[JobQueue] Error registering handlers: {e}")
        import traceback
        traceback.print_exc()

    # Register probability handlers
    try:
        from core.probability.probability_handler import register_probability_handlers
        register_probability_handlers(job_queue)
        print(f"[JobQueue] Probability handlers registered")
    except Exception as e:
        print(f"[JobQueue] Error registering probability handlers: {e}")
        import traceback
        traceback.print_exc()


def get_job_queue() -> JobQueue:
    """Get or create the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        queue_dir = Path("/mnt/data/job_queue")
        _job_queue = JobQueue(queue_dir, max_workers=20)
        _register_all_handlers(_job_queue)
        _job_queue.start_worker()
    else:
        # Ensure handlers are registered even if queue already exists
        # This handles cases where the singleton exists but handlers weren't registered
        if not _job_queue._handlers:
            _register_all_handlers(_job_queue)
    return _job_queue

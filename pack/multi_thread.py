import logging
import threading
from typing import Dict

from tqdm.auto import tqdm

# Configure basic logging with WARNING level
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ThreadProgressBar:
    """
    A class to manage multiple progress bars for different threads
    Each thread gets its own progress bar to track processing progress
    """
    def __init__(self, questions_per_thread: int):
        """
        Initialize progress bar manager
        Args:
            questions_per_thread: Number of items to process per thread
        """
        # Dictionary to store progress bars for each thread
        self.progress_bars: Dict[int, tqdm] = {}
        # Thread lock for synchronization
        self.lock = threading.Lock()
        self.questions_per_thread = questions_per_thread
        # Flag to track if progress bars are closed
        self._closed = False

    def get_progress_bar(self, thread_id: int) -> tqdm:
        """
        Get or create a progress bar for a specific thread
        Args:
            thread_id: ID of the thread requesting progress bar
        Returns:
            tqdm progress bar instance for the thread
        """
        # Return existing progress bar if already created
        if thread_id in self.progress_bars:
            return self.progress_bars[thread_id]

        # Create new progress bar with thread synchronization
        with self.lock:
            if thread_id not in self.progress_bars and not self._closed:
                # Create new progress bar with thread-specific position and description
                self.progress_bars[thread_id] = tqdm(total=self.questions_per_thread, desc=f"Thread-{thread_id}",
                                                     position=len(self.progress_bars) + 1, leave=True)
            return self.progress_bars[thread_id]

    def update_progress(self, thread_id: int) -> None:
        """
        Update progress for a specific thread's progress bar
        Args:
            thread_id: ID of the thread to update progress for
        """
        # Increment progress bar if it exists and isn't closed
        if thread_id in self.progress_bars and not self._closed:
            self.progress_bars[thread_id].update(1)

    def close_all(self) -> None:
        """
        Close all progress bars safely
        Handles cleanup of all progress bar instances
        """
        with self.lock:
            if not self._closed:
                # Attempt to close each progress bar
                for pbar in self.progress_bars.values():
                    try:
                        pbar.close()
                    except Exception as e:
                        logger.warning(f"Error closing progress bar: {e}")
                # Mark progress bars as closed
                self._closed = True
"""Definition of Timer
Adjusted from 
https://pypi.org/project/codetiming/
"""

# Standard library imports
import math
import time
from contextlib import ContextDecorator

from typing import Any, Callable, ClassVar, Optional, Union

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator"""

    def __init__(self, name):
        self.name = name
        self._start_time = None
        self._last = None
        self._checkpoints = []

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()
        self._last = self._start_time

    def elapsed(self, text) -> float:
        """Return how much time was spent so far, without stopping the timer"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        last = time.perf_counter()
        elapsed_ms = (last - self._last)*1e3
        self._last = last
        self._checkpoints.append((text, elapsed_ms))
        return elapsed_ms

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        last = time.perf_counter()
        total_msecs = (last - self._start_time)*1e3
        checkpoints = " ".join(map(lambda text_ms: f"[{text_ms[0]}={text_ms[1]:.1f}ms]", self._checkpoints))
        print (f"[{self.name}] {total_msecs:.1f}ms {checkpoints}")
        self._start_time = None
        return last

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()

if __name__ == "__main__":
    main ()

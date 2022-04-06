import time
import functools

import typing as t

from dataclasses import dataclass, field
from logger import getLogger

logger = getLogger('TRACE', filepath='timer')

class TimeError(Exception):
    def __init__(self, reason: str='An Unexpected Error has occured within the Time Module'):
        self.reason = reason

    def __str__(self):
        return self.reason

@dataclass
class FunctionTimer():
    timers = {}

    name: str = None
    text: str = "Elapsed Time: {:0.4f} seconds"
    _start: t.Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        ''' Add Timer to `self.timers` dictionary after initializiting the class '''
        if self.name is not None:
            self.timers.setdefault(self.name, 0)

    def start(self):
        ''' Start a Timer '''

        if self._start is not None:
            raise TimeError('Timer is already running')
        self._start = time.perf_counter()

    def stop(self):
        ''' Stop the Timer '''
        if self._start is None:
            raise TimeError('Timer is not running')

        elapsed = time.perf_counter() - self._start
        self._start = None

        return elapsed

    def __enter__(self):
        ''' For Use as a Context Manager '''

        self.start()

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        ''' For Use as a Context Manager '''

        self.stop()


def timed(f: t.Callable):
    ''' A Decorator for Timing Functions '''

    @functools.wraps(f)
    def wrapper(*args, **kwargs):

        start = time.perf_counter()
        _value = f(*args, **kwargs)
        end = time.perf_counter()

        logger.debug('timer', f"Elapsed time: {(end-start):0.4f} seconds")

        return _value

    return wrapper

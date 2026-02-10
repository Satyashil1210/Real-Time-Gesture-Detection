"""
Buffer utilities for RT-Gesture3D.

If you ever move from single-frame gestures to temporal gestures
(gestures defined over a short sequence), this module will help.
"""

from collections import deque
from typing import Deque, Generic, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """
    Fixed-size buffer (FIFO) that keeps only the last N items.

    Used for:
      - smoothing predictions over multiple frames
      - storing last N landmark sets or feature vectors
    """

    def __init__(self, maxlen: int):
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")

        self._buf: Deque[T] = deque(maxlen=maxlen)

    def append(self, item: T) -> None:
        self._buf.append(item)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def __iter__(self) -> Iterator[T]:
        return iter(self._buf)

    def to_list(self) -> list[T]:
        return list(self._buf)

    def latest(self) -> Optional[T]:
        return self._buf[-1] if self._buf else None

    def is_full(self) -> bool:
        return len(self._buf) == self._buf.maxlen

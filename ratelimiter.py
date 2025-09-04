import time, threading
from collections import deque

class RateLimiter:
    '''At most max_calls per period seconds'''

    def __init__(self, max_calls: int = 5, period: float = 1.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            while self.calls and (now - self.calls[0]) > self.period:
                self.calls.popleft()

            sleep_for = 0.0
            if len(self.calls) >= self.max_calls:
                sleep_for = self.period - (now - self.calls[0])
                
        if sleep_for > 0:
            time.sleep(sleep_for)
        with self.lock:
            self.calls.append(time.monotonic())

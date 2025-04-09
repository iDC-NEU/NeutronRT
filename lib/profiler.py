import time, logging

class Profiler():
    def __init__(self):
        self.last = {}
        self.logger = logging.getLogger('profiler')
        self.records = {}
        self.checkpoints = {}


    def begin(self, clock_id: int):
        self.last[clock_id] = time.perf_counter()


    def now(self, clock_id: int) -> float:
        return time.perf_counter() - self.last[clock_id]


    def dur(self, clock_id: int) -> float:
        last, now = self.last[clock_id], time.perf_counter()
        self.last[clock_id] = now
        return now - last
    

    def record(self, key: str, t: float):
        # self.logger.debug(f'record: {key} <- {t}')
        if self.records.get(key, None) is None:
            self.records[key] = t
        else:
            self.records[key] += t

        if self.checkpoints.get(key, None) is None:
            self.checkpoints[key] = t
        else:
            self.checkpoints[key] += t

import time


class Timer:
    """
    Usage as context manager: :

        with Timer() as timer:
            # ...

    Usage as class: :

        timer = Timer()
        # ...
        timer.stop()

    """

    def __init__(self, start=True):

        if start:
            self.start()

    def start(self):

        self.start_time = time.perf_counter()

    def stop(self, report=True):

        total_time = time.perf_counter() - self.start_time
        self.start_time = None

        if report:
            print(f'Total time: {total_time:.2f}s = {total_time/60:.2f}m')

        return total_time

    def __enter__(self):

        self.start()

    def __exit__(self, *_, **__):

        self.stop(report=True)

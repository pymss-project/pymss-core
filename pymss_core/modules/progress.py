class ProgressStepper:
    def __init__(self, emit, total_steps, start=0.0, end=1.0):
        self.emit = emit
        self.total_steps = max(1, int(total_steps))
        self.start = float(start)
        self.span = float(end) - float(start)
        self.step_index = 0

    def step(self, count=1):
        self.step_index = min(self.total_steps, self.step_index + int(count))
        self.emit(self.start + self.span * (self.step_index / self.total_steps))


def emit_progress_fraction(module, fraction):
    callback = getattr(module, "_pymss_progress_fraction_callback", None)
    if callback is not None:
        callback(float(fraction))


def progress_stepper(module, total_steps, start=0.0, end=1.0):
    return ProgressStepper(lambda fraction: emit_progress_fraction(module, fraction), total_steps, start, end)

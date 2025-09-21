class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best: float | None = None
        self.bad_epochs: int = 0

    def step(self, current: float) -> bool:
        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience

class TrainingIterationResults:
    def __init__(self, iterationLoss: float, iterationLatency: float) -> None:
        self.iterationLoss: float = iterationLoss
        self.iterationLatency: float = iterationLatency


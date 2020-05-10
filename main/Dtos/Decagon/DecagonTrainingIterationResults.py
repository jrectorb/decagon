from ..TrainingIterationResults import TrainingIterationResults

class DecagonTrainingIterationResults(TrainingIterationResults):
    def __init__(
        self,
        iterationLoss: float,
        iterationLatency: float,
        iterationEdgeType: tuple
    ) -> None:
        super().__init__(iterationLoss, iterationLatency)
        self.iterationEdgeType: tuple  = iterationEdgeType


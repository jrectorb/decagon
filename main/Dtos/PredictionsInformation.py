import numpy as np

class PredictionsInformation:
    def __init__(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        auroc: float,
        auprc: float,
        confusionMatrix: np.ndarray
    ) -> None:
        self.probabilities: np.ndarray = probabilities
        self.labels: np.ndarray = labels
        self.auroc: float = auroc
        self.auprc: float = auprc
        self.confusionMatrix: np.ndarray = confusionMatrix


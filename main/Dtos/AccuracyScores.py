class AccuracyScores:
    def __init__(self, auroc: float, auprc: float, aupk: float):
        self.auroc: float = auroc
        self.auprc: float = auprc
        self.aupk:  float = aupk


class AccuracyScores:
    def __init__(self, auroc: float, auprc: float, apk: float):
        self.auroc: float = auroc
        self.auprc: float = auprc
        self.apk:  float = apk


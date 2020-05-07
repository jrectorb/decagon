from .BaseLogger import BaseLogger
from ..Utils.Config import Config

class DecagonLogger(BaseLogger):
    '''
    Note that this class is not thread-safe
    '''

    def __init__(
        self,
        optimizer: Optimizer,
        placeholders: PlaceholdersDict,
        config: Config
    ) -> None:
        super().__init__(config)

        self.numItersPerCheckpoint: int = int(
            config.getSetting('NumIterationsPerCheckpoint')
        )

        self.trainResultLogDir: str = config.getSetting('TrainIterationResultDir')
        self.checkpointDir: str = config.getSetting('CheckpointDirectory')

        self.accuracyEvaluator: DecagonAccuracyEvaluator = DecagonAccuracyEvaluator(
            optimizer,
            placeholders,
            config
        )

    @property
    def _shouldCheckpoint(self):
        return (self.numIterationsDone % self.numItersPerCheckpoint) == 0

    def log(self, iterationResults: DecagonTrainingIterationResults) -> None:
        accuracyScores = self.accuracyEvaluator.evaluate(iterationResults)

        self._writeResultsToFileSystem(iterationResults, accuracyScores)
        self._writeResultsToStdout(iterationResults, accuracyScores)

        if self._shouldCheckpoint:
            self._checkpointModel()

        return

    def _writeResultsToFileSystem(
        self,
        iterationResults: DecagonTrainingIterationResults,
        accuracyScores: AccuracyScores
    ) -> None:
        pass

    def _writeResultsToStdout(
        self,
        iterationResults: DecagonTrainingIterationResults,
        accuracyScores: AccuracyScores
    ) -> None:
        pass


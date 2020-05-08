from .BaseLogger import BaseLogger
from ..Utils.Config import Config
import tensorflow as tf

class DecagonLogger(BaseLogger):
    '''
    Note that this class is not thread-safe
    '''

    def __init__(
        self,
        session: tf.Session,
        optimizer: Optimizer,
        placeholders: PlaceholdersDict,
        miniBatchIterator: EdgeMinibatchIterator,
        config: Config
    ) -> None:
        super().__init__(config)

        self.numItersPerCheckpoint: int = int(
            config.getSetting('NumIterationsPerCheckpoint')
        )

        self.trainResultLogDir: str = config.getSetting('TrainIterationResultDir')
        self.checkpointDir: str = config.getSetting('CheckpointDirectory')

        self.miniBatchIterator: EdgeMinibatchIterator = miniBatchIterator
        self.accuracyEvaluator: DecagonAccuracyEvaluator = DecagonAccuracyEvaluator(
            optimizer,
            placeholders,
            config
        )

    @property
    def _shouldCheckpoint(self):
        return (self.numIterationsDone % self.numItersPerCheckpoint) == 0

    def log(
        self,
        feedDict: FeedDict,
        iterationResults: DecagonTrainingIterationResults
    ) -> None:
        accuracyScores = self._computeAccuracyScores(feedDict)

        self._writeResultsToFileSystem(iterationResults, accuracyScores)
        self._writeResultsToStdout(iterationResults, accuracyScores)

        if self._shouldCheckpoint:
            self._checkpointModel()

        return

    def _computeAccuracyScores(self, feedDict: FeedDict) -> AccuracyScores:
        return self.accuracyEvaluator.evaluate(
            feedDict,
            self.miniBatchIterator.idx2edge_type[minibatch.current_edge_type_idx],
            self.miniBatchIterator.val_edges,
            self.miniBatchIterator.val_edges_false
        )

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


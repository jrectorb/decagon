from .BaseLogger import BaseLogger
from ..Utils.Config import Config
import tensorflow as tf
import atexit
import os

LOG_FILE_FORMAT = 'decagon_iteration_results_%d.csv'
PERC_IDX = LOG_FILE_FORMAT.find('%')
DOT_IDX  = LOG_FILE_FORMAT.find('.')

def _closeFile(f: file) -> None:
    f.close()

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

        self.trainResultLogFile: file = self._getTrainResultFile(config)
        self.trainResultWriter: DictWriter = self._getDictWriter()
        self.trainResultWriter.writeheader()

        self.checkpointDir: str = config.getSetting('CheckpointDirectory')

        self.miniBatchIterator: EdgeMinibatchIterator = miniBatchIterator
        self.accuracyEvaluator: DecagonAccuracyEvaluator = DecagonAccuracyEvaluator(
            optimizer,
            placeholders,
            config
        )

        atexit.register(_closeFile, f=self.trainResultLogFile)

    def _getTrainResultFile(self, config: Config) -> file:
        return open(self._getTrainResultFileName(config), 'w')

    def _getTrainResultFileName(self, config: Config) -> str:
        baseDir = config.getSetting('TrainIterationResultDir')
        existingIndices = [
            self._getFnameIdx(f)
            for f in os.listdir(baseDir)
            if self._isValidFname(baseDir, f)
        ]

        thisFileIdx = 0
        if len(existingIndices) > 0:
            thisFileIdx = max(thisFileIdx)

        return LOG_FILE_FORMAT % thisFileIdx

    def _getFnameIdx(self, fname: str) -> int:
        stripPre = fname.lstrip(LOG_FILE_FORMAT[:PERC_IDX])
        stripPost = stripPre.rstrip(LOG_FILE_FORMAT[DOT_IDX:])

        return int(stripPost)

    def _isValidFname(self, baseDir: str, fname: str) -> bool:
        isFile = os.isfile(baseDir, fname)
        isGoodPrefix = fname[:PERC_IDX] == LOG_FILE_FORMAT[:PERC_IDX]
        isGoodPostfix = fname[DOT_IDX:] = LOG_FILE_FORMAT[DOT_IDX:]

        return isFile and isGoodPrefix and isGoodPostfix

    def _getDictWriter(self) -> csv.DictWriter:
        fieldnames = [
            'IterationNum',
            'Loss',
            'Latency',
            'EdgeType',
            'AUROC',
            'AUPRC',
            'APK'
        ]

        return csv.DictWriter(self.trainResultLogFile, fieldnames=fieldnames)

    @property
    def _shouldCheckpoint(self):
        return (self.numIterationsDone % self.numItersPerCheckpoint) == 0

    def log(
        self,
        feedDict: FeedDict,
        iterationResults: DecagonTrainingIterationResults
    ) -> None:
        accuracyScores = self._computeAccuracyScores(feedDict)

        iterRowDict = self._getCsvRowDict(iterationResults, accuracyScores)
        iterString  = self._getString(iterationResults, accuracyScores)

        self.trainResultWriter.writerow(rowDict)
        print(iterString)

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

    def _getCsvRowDict(
        self,
        iterationResults: DecagonTrainingIterationResults,
        accuracyScores: AccuracyScores
    ) -> Dict:
        return {
            'IterationNum': self.numIterationsDone,
            'Loss': iterationResults.iterationLoss,
            'Latency': iterationResults.iterationLatency,
            'EdgeType': iterationRresults.iterationEdgeType,
            'AUROC': accuracyScores.auroc,
            'AUPRC': accuracyScores.auprc,
            'APK': accuracyScores.apk,
        }

    def _getString(
        self,
        iterationResults: DecagonTrainingIterationResults,
        accuracyScores: AccuracyScores
    ) -> str:
        '''
            IterationNum: %d
            Loss: %f
            Latency: %f
            EdgeType: %s
            AUROC: %f
            AUPRC: %f
            APK: %f



        ''' % (
            self.numIterationsDone,
            iterationResults.iterationLoss,
            iterationResults.iterationLatency,
            iterationRresults.iterationEdgeType,
            accuracyScores.auroc,
            accuracyScores.auprc,
            accuracyScores.apk,
        )


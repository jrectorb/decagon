from .BaseLogger import BaseLogger
from ..Checkpointer.TensorflowCheckpointer import TensorflowCheckpointer
from ..Dtos.Enums.DataSetType import DataSetType
from ..Utils.Config import Config
import _io
import tensorflow as tf
import atexit
import os

LOG_FILE_FORMAT = 'decagon_iteration_results_%d.csv'
PERC_IDX = LOG_FILE_FORMAT.find('%')
DOT_IDX  = LOG_FILE_FORMAT.find('.')

def _closeFile(f: _io.TextIOWrapper) -> None:
    f.close()

class DecagonLogger(BaseLogger, dataSetType=None):
    '''
    Note that this class is not thread-safe
    '''
    def __init__(
        self,
        session: tf.Session,
        trainable: DecagonTrainable,
        checkpointer: TensorflowCheckpointer,
        config: Config
    ) -> None:
        super().__init__(config)

        self.trainResultLogFile: _io.TextIOWrapper = self._getTrainResultFile(config)
        self.trainResultWriter: DictWriter = self._getDictWriter()
        self.trainResultWriter.writeheader()

        self.checkpointer: TensorflowCheckpointer = checkpointer

        self.trainable: DecagonTrainable = trainable
        self.accuracyEvaluator: DecagonAccuracyEvaluator = DecagonAccuracyEvaluator(
            trainable.optimizer,
            trainable.placeholders,
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
    def shouldLog(self):
        return super().shouldLog or self.checkpointer.shouldCheckpoint

    def incrementIterations(self) -> None:
        super().incrementIterations()
        self.checkpointer.incrementIterations()

    def log(
        self,
        feedDict: FeedDict,
        iterationResults: DecagonTrainingIterationResults
    ) -> None:
        if super().shouldLog:
            self._logInternal(feedDict, iterationResults)

        if self.checkpointer.shouldCheckpoint:
            self.checkpointer.save()

        return

    # Force log to filesystem and stdout at epoch end
    def logEpochEnd(
        self,
        feedDict: FeedDict,
        iterationResults: DecagonTrainingIterationResults
    ) -> None:
        self._logInternal(feedDict, iterationResults)
        self.checkpointer.save()

    def _logInternal(
        self,
        feedDict: FeedDict,
        iterationResults: DecagonTrainingIterationResults
    ) -> None:
        accuracyScores = self._computeAccuracyScores(feedDict)

        iterRowDict = self._getCsvRowDict(iterationResults, accuracyScores)
        iterString  = self._getString(iterationResults, accuracyScores)

        self.trainResultWriter.writerow(rowDict)
        print(iterString)

        return

    def _computeAccuracyScores(self, feedDict: FeedDict) -> AccuracyScores:
        iterator = self.trainable.dataSetIterator

        return self.accuracyEvaluator.evaluate(
            feedDict,
            iterator.idx2edge_type[iterator.current_edge_type_idx],
            iterator.val_edges,
            iterator.val_edges_false
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
            iterationResults.iterationEdgeType,
            accuracyScores.auroc,
            accuracyScores.auprc,
            accuracyScores.apk,
        )

class DecagonDummyDataLogger(BaseLogger, dataSetType=DataSetType.DecagonDummyData):
    pass

class DecagonPublicDataLogger(BaseLogger, dataSetType=DataSetType.DecagonPublicData):
    pass


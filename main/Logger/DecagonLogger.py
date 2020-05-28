from .BaseLogger import BaseLogger
from ..AccuracyEvaluators.Tensorflow.DecagonAccuracyEvaluator import DecagonAccuracyEvaluator
from ..Checkpointer.TensorflowCheckpointer import TensorflowCheckpointer
from ..Dtos.AccuracyScores import AccuracyScores
from ..Dtos.Decagon.DecagonTrainingIterationResults import DecagonTrainingIterationResults
from ..Dtos.Enums.LoggerType import LoggerType
from ..Trainable.Decagon.DecagonTrainable import DecagonTrainable
from ..Utils.Config import Config
from pathlib import Path
from typing import Dict
import _io
import tensorflow as tf
import numpy as np
import atexit
import csv
import os

LOG_FILE_FORMAT = 'decagon_iteration_results_%d.csv'
PERC_IDX = LOG_FILE_FORMAT.find('%')
DOT_IDX  = LOG_FILE_FORMAT.find('.')

def _closeFile(f: _io.TextIOWrapper) -> None:
    f.close()

class DecagonLogger(BaseLogger, functionalityType=LoggerType.DecagonLogger):
    '''
    Note that this class is not thread-safe
    '''
    def __init__(
        self,
        session: tf.Session,
        dataSetId: str,
        trainable: DecagonTrainable,
        checkpointer: TensorflowCheckpointer,
        config: Config
    ) -> None:
        super().__init__(config)

        self.session: tf.Session = session
        self.dataSetId: str = dataSetId
        self.currEpoch: int = 1

        self.trainResultLogFile: _io.TextIOWrapper = self._getTrainResultFile(config)
        self.trainResultWriter: DictWriter = self._getDictWriter()
        self.trainResultWriter.writeheader()

        self.checkpointer: TensorflowCheckpointer = checkpointer

        self.writeNdarrays = bool(config.getSetting('WriteNdarrays'))
        if self.writeNdarrays:
            self.ndarrayWritePath = config.getSetting('NdarrayWriteDir')
            Path(self.ndarrayWritePath).mkdir(parents=True, exist_ok=True)

        self.trainable: DecagonTrainable = trainable
        self.accuracyEvaluator: DecagonAccuracyEvaluator = DecagonAccuracyEvaluator(
            self.session,
            trainable.placeholders,
            trainable.optimizer.predictions,
            trainable.dataSetIterator.edge_type2idx,
            config
        )

        atexit.register(_closeFile, f=self.trainResultLogFile)

    def _getTrainResultFile(self, config: Config) -> _io.TextIOWrapper:
        return open(self._getTrainResultFileName(config), 'w')

    def _getTrainResultFileName(self, config: Config) -> str:
        baseDir = config.getSetting('TrainIterationResultDir')
        Path(baseDir).mkdir(parents=True, exist_ok=True)

        existingIndices = [
            self._getFnameIdx(f)
            for f in os.listdir(baseDir)
            if self._isValidFname(baseDir, f)
        ]

        thisFileIdx = 0
        if len(existingIndices) > 0:
            thisFileIdx = max(existingIndices) + 1

        return baseDir + LOG_FILE_FORMAT % thisFileIdx

    def _getFnameIdx(self, fname: str) -> int:
        stripPre = fname.lstrip(LOG_FILE_FORMAT[:PERC_IDX])
        stripPost = stripPre.rstrip(LOG_FILE_FORMAT[DOT_IDX:])

        return int(stripPost)

    def _isValidFname(self, baseDir: str, fname: str) -> bool:
        isFile = os.path.isfile(baseDir + fname)
        isGoodPrefix = fname[:PERC_IDX] == LOG_FILE_FORMAT[:PERC_IDX]
        isGoodPostfix = fname[fname.rfind('.'):] == LOG_FILE_FORMAT[DOT_IDX:]

        return isFile and isGoodPrefix and isGoodPostfix

    def _getDictWriter(self) -> csv.DictWriter:
        fieldnames = [
            'DataSetId',
            'Epoch',
            'IterationNum',
            'Loss',
            'Latency',
            'EvaluateAll',
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
        feedDict: Dict,
        iterationResults: DecagonTrainingIterationResults
    ) -> None:
        if super().shouldLog:
            self._logInternal(feedDict, iterationResults)

        if self.checkpointer.shouldCheckpoint:
            self.checkpointer.save()
            self._writeAsNdarray(feedDict)

        return

    # Force log to filesystem and stdout at epoch end
    def logEpochEnd(
        self,
        feedDict: Dict,
        iterationResults: DecagonTrainingIterationResults
    ) -> None:
        self._logInternal(feedDict, iterationResults, evalAll=True)
        self.checkpointer.save()

        self.currEpoch += 1

    def _logInternal(
        self,
        feedDict: Dict,
        iterationResults: DecagonTrainingIterationResults,
        evalAll: bool = False
    ) -> None:
        accuracyScores = self._computeAccuracyScores(feedDict, evalAll)

        iterRowDict = self._getCsvRowDict(iterationResults, accuracyScores, evalAll)
        iterString  = self._getString(iterationResults, accuracyScores, evalAll)

        self.trainResultWriter.writerow(iterRowDict)
        self.trainResultLogFile.flush()

        print(iterString)

        return

    def _computeAccuracyScores(self, feedDict: Dict, evalAll: bool) -> AccuracyScores:
        iterator = self.trainable.dataSetIterator

        if evalAll:
            return self.accuracyEvaluator.evaluateAll(
                feedDict,
                iterator.val_edges,
                iterator.val_edges_false
            )

        else:
            return self.accuracyEvaluator.evaluate(
                feedDict,
                (1, 1, 0),
                iterator.val_edges,
                iterator.val_edges_false
            )

    def _getCsvRowDict(
        self,
        iterationResults: DecagonTrainingIterationResults,
        accuracyScores: AccuracyScores,
        evalAll: bool
    ) -> Dict:
        return {
            'DataSetId': self.dataSetId,
            'Epoch': self.currEpoch,
            'EvaluateAll': evalAll,
            'IterationNum': self.numIterationsDone,
            'Loss': iterationResults.iterationLoss,
            'Latency': iterationResults.iterationLatency,
            'EdgeType': iterationResults.iterationEdgeType,
            'AUROC': accuracyScores.auroc,
            'AUPRC': accuracyScores.auprc,
            'APK': accuracyScores.apk,
        }

    def _getString(
        self,
        iterationResults: DecagonTrainingIterationResults,
        accuracyScores: AccuracyScores,
        evalAll: bool
    ) -> str:
        return '''
DataSetId: %s
Epoch: %d
IterationNum: %d
Loss: %f
Latency: %f
Evaluated All: %r
EdgeType: %s
AUROC: %f
AUPRC: %f
APK: %f

        ''' % (
            self.dataSetId,
            self.currEpoch,
            self.numIterationsDone,
            iterationResults.iterationLoss,
            iterationResults.iterationLatency,
            evalAll,
            iterationResults.iterationEdgeType,
            accuracyScores.auroc,
            accuracyScores.auprc,
            accuracyScores.apk,
        )

    def _writeAsNdarray(self, feedDict) -> None:
        self._writeEmbeddings(feedDict)
        self._writeRelationEmbeddingImportance(feedDict)
        self._writeRelationMtx(feedDict)

    def _writeEmbeddings(self, feedDict) -> None:
        DRUG_GRAPH_IDX = 1
        embeddingsToWrite = self.session.run(
            self.trainable.model.embeddings[DRUG_GRAPH_IDX],
            feed_dict=feedDict
        )

        np.save(
            self.ndarrayWritePath + 'embeddings.npy',
            embeddingsToWrite,
            allow_pickle=False
        )

    def _writeRelationEmbeddingImportance(self, feedDict) -> None:
        validIdxs = [
            i for i in range(len(self.trainable.dataSetIterator.idx2edge_type))
            if self._edgeTypeValid(i)
        ]

        embeddingImportanceMtxTensors = [
            self.trainable.model.latent_varies[idx]
            for idx in validIdxs
        ]

        embeddingImportanceMtx = self.session.run(
            embeddingImportanceMtxTensors,
            feed_dict=feedDict
        )

        np.savez(
            self.ndarrayWritePath + 'EmbeddingImportance.npyz',
            embeddingImportanceMtx,
            allow_pickle=False
        )

    def _writeRelationMtx(self, feedDict) -> None:
        globInterIdx = next((
            i for i in range(len(self.trainable.dataSetIterator.idx2edge_type))
            if self._edgeTypeValid(i)
        ))

        globalRelationMtx = self.session.run(
            self.trainable.model.latent_inters[globInterIdx],
            feed_dict=feedDict
        )

        np.save(
            self.ndarrayWritePath + 'GlobalRelations.npy',
            globalRelationMtx,
            allow_pickle=False
        )

    def _edgeTypeValid(self, idx: int) -> bool:
        DRUG_DRUG_GRAPH_TUPLE = (1, 1)
        return self.trainable.dataSetIterator.idx2edge_type[idx][:2] == DRUG_DRUG_GRAPH_TUPLE


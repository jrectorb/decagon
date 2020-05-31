from ..ActiveLearner.BaseActiveLearner import BaseActiveLearner
from ..DataSetParsers.DataSetBuilder import DataSetBuilder
from ..Trainable.BaseTrainableBuilder import BaseTrainableBuilder
from ..Trainer.BaseTrainer import BaseTrainer
from ..Dtos.DataSet import DataSet
from ..Dtos.Enums.ActiveLearnerType import ActiveLearnerType
from ..Dtos.Enums.TrainableType import TrainableType
from ..Dtos.Enums.TrainerType import TrainerType
from ..Dtos.IterationResults import IterationResults
from ..Dtos.NodeIds import SideEffectId
from ..Dtos.Trainable import Trainable
from ..Utils.ArgParser import ArgParser
from ..Utils.Config import Config
from ..Utils.ObjectFactory import ObjectFactory

from typing import Type, Dict
from pathlib import Path
import numpy as np
import tensorflow as tf
import ray
import sys
import os

class CheckpointToNdarrayWriter:
    def __init__(self, ckptDir: str, ndarrayBaseDir: str) -> None:
        self.ckptDir: str = ckptDir
        self.ndarrayBaseDir: str = ndarrayBaseDir
        Path(self.ndarrayBaseDir).mkdir(parents=True, exist_ok=True)

    def write(self) -> None:
        print('Beginning write')

        trainable: Type[Trainable] = self._getDecagonTrainable()
        populatedSession: tf.Session = self._getPopulatedSession()

        feedDict: Dict = self._getFeedDict(trainable)
        self._writeAsNdarray(trainable, populatedSession, feedDict)

        print('Finished writing as ndarrays to %s' % self.ndarrayBaseDir)

    def _getDecagonTrainable(self) -> Type[Trainable]:
        config: Config = self._getConfig()
        self._setEnvVars(config)
        dataSet: Type[DataSet] = DataSetBuilder.build(config)

        self._setSideEffectIdx(dataSet)

        return self._getTrainable(dataSet, config)

    def _setSideEffectIdx(self, dataSet) -> None:
        self.sideEffectIdx = [
            SideEffectId.toDecagonFormat(raw)
            for raw in dataSet.adjacencyMatrices.drugDrugRelationMtxs.keys()
        ]

    def _getConfig(self) -> Config:
        argParser = ArgParser()
        argParser.parse()

        return Config(argParser)

    def _setEnvVars(self, config: Config) -> None:
        if bool(config.getSetting('UseGpu')):
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def _getTrainable(
        self,
        dataSet: Type[DataSet],
        config: Config
    ) -> Type[Trainable]:
        # Don't need to actually run with test edges here, so just
        # pass an empty dict
        trainableBuilder = ObjectFactory.build(
            BaseTrainableBuilder,
            TrainableType.DecagonTrainable,
            dataSet=dataSet,
            drugDrugTestEdges={},
            config=config,
        )

        return trainableBuilder.build()

    def _getPopulatedSession(self):
        session = tf.Session()

        ckptPath = tf.train.latest_checkpoint(self.ckptDir)
        tf.train.Saver().restore(session, ckptPath)

        return session

    def _getFeedDict(self, trainable) -> Dict:
        result = {}

        trainable.dataSetIterator.update_feed_dict(
            result,
            dropout=0.0,
            placeholders=trainable.model.placeholders
        )

        return result

    def _writeAsNdarray(self, decTrainable, session, feedDict) -> None:
        self._writeEmbeddings(decTrainable, session, feedDict)
        self._writeRelationEmbeddingImportance(decTrainable, session, feedDict)
        self._writeRelationMtx(decTrainable, session, feedDict)

    def _writeEmbeddings(self, trainable, session, feedDict) -> None:
        DRUG_GRAPH_IDX = 1
        embeddingsToWrite = session.run(
            trainable.model.embeddings[DRUG_GRAPH_IDX],
            feed_dict=feedDict
        )

        np.save(
            self.ndarrayBaseDir + 'embeddings.npy',
            embeddingsToWrite,
            allow_pickle=False
        )

    def _writeRelationEmbeddingImportance(self, trainable, session, feedDict) -> None:
        validIdxs = [
            i for i in range(len(trainable.dataSetIterator.idx2edge_type))
            if self._edgeTypeValid(trainable, i)
        ]

        embeddingImportanceMtxTensors = [
            trainable.model.latent_varies[idx]
            for idx in validIdxs
        ]

        embeddingImportanceMtxs = session.run(
            embeddingImportanceMtxTensors,
            feed_dict=feedDict
        )

        for idx, importanceMtx in enumerate(embeddingImportanceMtxs):
            tposeStr = ''
            if idx >= len(self.sideEffectIdx):
                idx -= len(self.sideEffectIdx)
                tposeStr = '-Transposed'

            sideEffectIdStr = '%s%s' % (self.sideEffectIdx[idx], tposeStr)

            fname = 'EmbeddingImportance-%s.npy' % sideEffectIdStr
            np.save(
                self.ndarrayBaseDir + fname,
                importanceMtx,
                allow_pickle=False
            )

    def _writeRelationMtx(self, trainable, session, feedDict) -> None:
        globInterIdx = next((
            i for i in range(len(trainable.dataSetIterator.idx2edge_type))
            if self._edgeTypeValid(trainable, i)
        ))

        globalRelationMtx = session.run(
            trainable.model.latent_inters[globInterIdx],
            feed_dict=feedDict
        )

        np.save(
            self.ndarrayBaseDir + 'GlobalRelations.npy',
            globalRelationMtx,
            allow_pickle=False
        )

    def _edgeTypeValid(self, trainable, idx: int) -> bool:
        DRUG_DRUG_GRAPH_TUPLE = (1, 1)
        return trainable.dataSetIterator.idx2edge_type[idx][:2] == DRUG_DRUG_GRAPH_TUPLE

if __name__ == '__main__':
    writer = CheckpointToNdarrayWriter(
        '/home/mkkr/scratch/Projects/11_decagon/ckpts_rel_masking_more',
        '/home/mkkr/scratch/Projects/11_decagon/rel-masking-more-ndarray-from-ckpts/'
    )

    writer.write()


from .BaseTrainer import BaseTrainer
from ..Checkpointer.TensorflowCheckpointer import TensorflowCheckpointer
from ..Dtos.Decagon.DecagonTrainingIterationResults import DecagonTrainingIterationResults
from ..Dtos.Enums.TrainerType import TrainerType
from ..Dtos.Trainable import Trainable
from ..Logger.DecagonLogger import DecagonLogger
from ..Trainable.Decagon.DecagonTrainable import DecagonTrainable
from ..Utils.Config import Config
from typing import Dict
import time
import multiprocessing
import tensorflow as tf

class BaseDecagonTrainer(BaseTrainer, functionalityType=TrainerType.DecagonTrainer):
    def __init__(self, dataSetId: str, trainable: DecagonTrainable, config: Config) -> None:
        self.optimizer = trainable.optimizer
        self.dataSetIterator = trainable.dataSetIterator
        self.placeholders = trainable.placeholders

        tfConf = self._getTfConf()
        self.session: tf.Session = tf.Session(config=tfConf)

        checkpointer = TensorflowCheckpointer(trainable, self.session, config)
        self.logger = DecagonLogger(
            self.session,
            dataSetId,
            trainable,
            checkpointer,
            config
        )

        self.numEpochs: int = int(config.getSetting('NumEpochs'))
        self.dropoutRate: float = float(config.getSetting('dropout'))

    def _getTfConf(self) -> tf.ConfigProto:
        numThreads = multiprocessing.cpu_count()

        result = tf.ConfigProto()
        result.intra_op_parallelism_threads = numThreads
        result.inter_op_parallelism_threads = numThreads

        return result

    def train(self):
        self.session.run(tf.global_variables_initializer())

        feedDict = None
        for epochNum in range(self.numEpochs):
            self.dataSetIterator.shuffle()
            while not self.dataSetIterator.end():
                feedDict = self._getNextFeedDict()
                iterResults = self._trainBatch(feedDict)

                self.logger.incrementIterations()
                if self.logger.shouldLog:
                    self.logger.log(feedDict, iterResults)

            self.logger.logEpochEnd(feedDict, iterResults)

        return

    def _trainBatch(self, feedDict: Dict) -> DecagonTrainingIterationResults:
        tic = time.time()
        iterationLoss, iterationEdgeType = self._doIterationTraining(feedDict)
        toc = time.time()

        return DecagonTrainingIterationResults(
            iterationLoss,
            toc - tic,
            iterationEdgeType
        )

    def _getNextFeedDict(self) -> Dict:
        '''
        In the result dict, keys are tf.placeholder objects while result
        types may vary
        '''
        preResult = self.dataSetIterator.next_minibatch_feed_dict(self.placeholders)

        return self.dataSetIterator.update_feed_dict(
            preResult,
            self.dropoutRate,
            self.placeholders
        )

    def _doIterationTraining(self, feedDict: Dict) -> tuple:
        ITER_TRAIN_LOSS_IDX = 1
        ITER_EDGE_TYPE_IDX  = 2

        operations = [
            self.optimizer.opt_op,
            self.optimizer.cost,
            self.optimizer.batch_edge_type_idx,
        ]

        outputs = self.session.run(operations, feed_dict=feedDict)

        return outputs[ITER_TRAIN_LOSS_IDX], outputs[ITER_EDGE_TYPE_IDX]


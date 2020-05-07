from .BaseTrainer import BaseTrainer
from ..Dtos.Decagon.DecagonTrainingIterationResults import DecagonTrainingIterationResults
from ..Trainable import Trainable
from ..Utils.Config import Config
import time
import tensorflow as tf

class BaseDecagonTrainer(BaseTrainer):
    def __init__(self, trainable: DecagonTrainable, config: Config) -> None:
        self.optimizer = trainable.optimizer
        self.dataSetIterator = trainable.dataSetIterator
        self.placeholders = trainable.placeholders

        self.logger = DecagonLogger(config)
        self.numEpochs: int = int(config.getSetting('NumEpochs'))
        self.dropoutRate: float = float(config.getSetting('dropout'))

    def train(self):
        session = tf.Session()
        session.run(tf.global_variables_initializer())

        for epochNum in range(self.numEpochs):
            self.dataSetIterator.shuffle()
            while not self.dataSetIterator.end():
                iterResults = self._trainBatch(session)

                self.logger.incrementIterations()
                if self.logger.shouldLog:
                    self.logger.log(iterResults)

            self.logger.logEpochEnd(iterResults)

        self.logger.markTrainingEnd()

        return

    def _trainBatch(self, session: tf.Session) -> DecagonTrainingIterationResults:
        feedDict = self._getNextFeedDict()

        tic = time.time()
        iterationLoss, iterationEdgeType = self._doIterationTraining(session, feedDict)
        toc = time.time()

        return DecagonTrainingIterationResults(
            iterationLoss,
            toc - tic,
            iterationEdgeType
        )

    def _getNextFeedDict(self) -> Dict:
        preResult = self.dataSetIterator.next_minibatch_feed_dict(self.placeholders)

        return self.dataSetIterator.update_feed_dict(
            preResult,
            self.dropoutRate,
            self.placeholders
        )

    def _doIterationTraining(self, session: tf.Session, feedDict: Dict) -> tuple:
        ITER_TRAIN_LOSS_IDX = 1
        ITER_EDGE_TYPE_IDX  = 2

        operations = [
            self.optimizer.opt_op,
            self.optimizer.cost,
            self.optimizer.batch_edge_type_idx,
        ]

        outputs = session.run(operations, feedDict=feedDict)

        return outputs[ITER_TRAIN_LOSS_IDX], outputs[ITER_EDGE_TYPE_IDX]


class DummyDatasetDecagonTrainer(
    BaseDecagonTrainer,
    dataSetType = DataSetType.DecagonDummyData
):
    pass

class PublicDatasetDecagonTrainer(
    BaseDecagonTrainer,
    dataSetType = DataSetType.DecagonPublicData
):
    pass


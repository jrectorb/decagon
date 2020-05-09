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

        self.session: tf.Session = tf.Session()

        checkpointer = TensorflowCheckpointer(trainable, config)
        self.logger = DecagonLogger(self.session, trainable, checkpointer, config)

        self.numEpochs: int = int(config.getSetting('NumEpochs'))
        self.dropoutRate: float = float(config.getSetting('dropout'))

    def train(self):
        self.session.run(tf.global_variables_initializer())

        feedDict = None
        for epochNum in range(self.numEpochs):
            self.dataSetIterator.shuffle()
            while not self.dataSetIterator.end():
                feedDict = self._getNextFeedDict()
                iterResults = self._trainBatch(session, feedDict)

                self.logger.incrementIterations()
                if self.logger.shouldLog:
                    self.logger.log(iterResults, feedDict)

            self.logger.logEpochEnd(iterResults, feedDict)

        return

    def _trainBatch(self, feedDict: FeedDict) -> DecagonTrainingIterationResults:
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

        outputs = self.session.run(operations, feedDict=feedDict)

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


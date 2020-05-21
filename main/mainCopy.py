from .ActiveLearner.BaseActiveLearner import BaseActiveLearner
from .DataSetParsers.DataSetBuilder import DataSetBuilder
from .Trainable.BaseTrainableBuilder import BaseTrainableBuilder
from .Trainer.BaseTrainer import BaseTrainer
from .Dtos.DataSet import DataSet
from .Dtos.Enums.ActiveLearnerType import ActiveLearnerType
from .Dtos.Enums.TrainableType import TrainableType
from .Dtos.Enums.TrainerType import TrainerType
from .Dtos.IterationResults import IterationResults
from .Dtos.Trainable import Trainable
from .Utils.ArgParser import ArgParser
from .Utils.Config import Config
from .Utils.ObjectFactory import ObjectFactory

from typing import Type
import ray
import sys
import os

def _getConfig() -> Config:
    argParser = ArgParser()
    argParser.parse()

    return Config(argParser)

def _setEnvVars(config: Config) -> None:
    if bool(config.getSetting('UseGpu')):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

def _getTrainable(dataSet: Type[DataSet], config: Config) -> Type[Trainable]:
    trainableBuilder = ObjectFactory.build(
        BaseTrainableBuilder,
        TrainableType[config.getSetting('TrainableType')],
        dataSet=dataSet,
        config=config,
    )

    return trainableBuilder.build()

def _getActiveLearner(dataSet, config: Config) -> Type[BaseActiveLearner]:
    activeLearnerType = ActiveLearnerType[config.getSetting('ActiveLearnerType')]

    return ObjectFactory.build(
        BaseActiveLearner,
        activeLearnerType,
        initDataSet=dataSet,
        config=config
    )

def _getTrainer(
    dataSetId: str,
    trainable: Type[Trainable],
    config: Config
) -> Type[BaseTrainer]:
    trainerType = TrainerType[config.getSetting('TrainerType')]

    return ObjectFactory.build(
        BaseTrainer,
        trainerType,
        dataSetId=dataSetId,
        trainable=trainable,
        config=config
    )

def main() -> int:
    config: Config = _getConfig()
    _setEnvVars(config)
    dataSet: Type[DataSet] = DataSetBuilder.build(config)

    activeLearner: Type[BaseActiveLearner] = _getActiveLearner(dataSet, config)

    iterResults: Type[IterationResults] = None
    runningJobs = []
    while activeLearner.hasUpdate(dataSet, iterResults):
        dataSet = activeLearner.getUpdate(dataSet, iterResults)

        trainable: Type[Trainable] = _getTrainable(dataSet, config)
        trainer: Type[BaseTrainer] = _getTrainer(dataSet.id, trainable, config)

        trainer.train()

    ray.get(trainingJobs)

    return 0

def _hackyMain() -> int:
    from .DataSetParsers.AdjacencyMatrices.AnosmiaAdjMtxBuilder import AnosmiaAdjMtxBuilder
    from .DataSetParsers.AdjacencyMatrices.HyperglycaemiaAdjMtxBuilder import HyperglycaemiaAdjMtxBuilder
    from .DataSetParsers.AdjacencyMatrices.NeutropeniaAdjMtxBuilder import NeutropeniaAdjMtxBuilder

    ray.init()#local_mode=True)

    config: Config = _getConfig()
    _setEnvVars(config)

    adjMtxTypes = [
        #AnosmiaAdjMtxBuilder,
        HyperglycaemiaAdjMtxBuilder,
        #NeutropeniaAdjMtxBuilder,
    ]

    jobs = []

    #import pdb; pdb.set_trace()
    for adjMtxType in adjMtxTypes:
        dataSet: Type[DataSet] = DataSetBuilder.buildForMtxType(adjMtxType, config)
        activeLearner: Type[BaseActiveLearner] = _getActiveLearner(dataSet, config)

        #for _ in range(8):
        for _ in range(1):
            dataSet = activeLearner.getUpdate(dataSet, None)
            jobs.append(_doTraining.remote(dataSet, config))

    ray.get(jobs)

    return 0

def _newHackyMain() -> int:
    from .DataSetParsers.AdjacencyMatrices.AnosmiaAdjMtxBuilder import AnosmiaAdjMtxBuilder
    from .DataSetParsers.AdjacencyMatrices.HyperglycaemiaAdjMtxBuilder import HyperglycaemiaAdjMtxBuilder
    from .DataSetParsers.AdjacencyMatrices.NeutropeniaAdjMtxBuilder import NeutropeniaAdjMtxBuilder

    ray.init()#local_mode=True)

    config: Config = _getConfig()
    _setEnvVars(config)

    adjMtxTypes = [
        AnosmiaAdjMtxBuilder,
        HyperglycaemiaAdjMtxBuilder,
        NeutropeniaAdjMtxBuilder,
    ]

    objs = [
        _doTrainingGreedy.remote(adjMtxType, config)
        for adjMtxType in adjMtxTypes
    ]

    #from .ActiveLearner.GreedyActiveLearner import GreedyActiveLearner
    #for adjMtxType in adjMtxTypes:
    #    dataSet: Type[DataSet] = DataSetBuilder.buildForMtxType(adjMtxType, config)
    #    activeLearner: Type[BaseActiveLearner] = GreedyActiveLearner(dataSet, config)

    #    trainable = None
    #    trainer = None
    #    predTensor = None
    #    lastUsedFeedDict = None
    #    for _ in range(7):
    #        dataSet = activeLearner.getUpdate(
    #            predTensor,
    #            trainable.optimizer.placeholders if trainable else None,
    #            lastUsedFeedDict,
    #            trainer.session if trainer else None,
    #            dataSet,
    #            None
    #        )

    #        trainable: Type[Trainable] = _getTrainable(dataSet, config)
    #        trainer: Type[BaseTrainer] = _getTrainer(dataSet.id, trainable, config)

    #        lastUsedFeedDict = trainer.train()
    #        predTensor = trainable.optimizer.predictions

    ray.get(objs)

    return 0

@ray.remote(num_gpus=1, max_calls=1)
def _doTrainingGreedy(adjMtxType, config):
    from .ActiveLearner.GreedyActiveLearner import GreedyActiveLearner
    dataSet: Type[DataSet] = DataSetBuilder.buildForMtxType(adjMtxType, config)
    activeLearner: Type[BaseActiveLearner] = GreedyActiveLearner(dataSet, config)

    trainable = None
    trainer = None
    predTensor = None
    lastUsedFeedDict = None
    for _ in range(8):
        dataSet = activeLearner.getUpdate(
            predTensor,
            trainable.optimizer.placeholders if trainable else None,
            lastUsedFeedDict,
            trainer.session if trainer else None,
            dataSet,
            None
        )

        trainable: Type[Trainable] = _getTrainable(dataSet, config)
        trainer: Type[BaseTrainer] = _getTrainer(dataSet.id, trainable, config)

        lastUsedFeedDict = trainer.train()
        predTensor = trainable.optimizer.predictions

    return

@ray.remote(num_gpus=1, max_calls=1)
def _doTraining(dataSet: DataSet, config: Config):
    trainable: Type[Trainable] = _getTrainable(dataSet, config)
    trainer: Type[BaseTrainer] = _getTrainer(dataSet.id, trainable, config)

    trainer.train()

    return

if __name__ == '__main__':
    sys.exit(_newHackyMain())



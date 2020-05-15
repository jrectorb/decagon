from .ActiveLearner.BaseActiveLearner import BaseActiveLearner
from .DataSetParsers.DataSetBuilder import DataSetBuilder
from .Trainable.BaseTrainableBuilder import BaseTrainableBuilder
from .Trainer.BaseTrainer import BaseTrainer
from .Dtos.DataSet import DataSet
from .Dtos.Enums.ActiveLearnerType import ActiveLearnerType
from .Dtos.Enums.TrainableType import TrainableType
from .Dtos.Enums.TrainerType import TrainerType
from .Dtos.IterationResults import IterationResults
from .Dtos.Trainable.Trainable import Trainable
from .Utils.ArgParser import ArgParser
from .Utils.Config import Config
from .Utils.ObjectFactory import ObjectFactory

from typing import Type
import sys

def _getConfig() -> Config:
    argParser = ArgParser()
    argParser.parse()

    return Config(argParser)

def _getTrainable(dataSet: Type[DataSet], config: Config) -> Type[Trainable]:
    trainableBuilder = ObjectFactory.build(
        BaseTrainableBuilder,
        TrainableType[config.getSetting('TrainableType')],
        dataSet=dataSet,
        config=config
    )

    return trainableBuilder.build()

def _getActiveLearner(config: Config) -> Type[BaseActiveLearner]:
    activeLearnerType = ActiveLearnerType[config.getSetting('ActiveLearnerType')]

    return ObjectFactory.build(
        BaseActiveLearner,
        activeLearnerType,
        config=config
    )

def _getTrainer(trainable: Type[Trainable], config: Config) -> Type[BaseTrainer]:
    trainerType = TrainerType[config.getSetting('TrainerType')]

    return ObjectFactory.build(
        BaseTrainer,
        trainerType,
        trainable=trainable,
        config=config
    )

def main() -> int:
    config: Config = _getConfig()
    dataSet: Type[DataSet] = DataSetBuilder.build(config)

    activeLearner: Type[BaseActiveLearner] = _getActiveLearner(config)

    iterResults: Type[IterationResults] = None
    while activeLearner.hasUpdate(dataSet, iterResults):
        dataSet = activeLearner.getUpdate(dataSet, iterResults)

        trainable: Type[Trainable] = _getTrainable(dataSet, config)
        trainer: Type[BaseTrainer] = _getTrainer(trainable, config)

        trainer.train()

        iterResults = trainable.getIterationResults()

    return 0

if __name__ == '__main__':
    sys.exit(main())


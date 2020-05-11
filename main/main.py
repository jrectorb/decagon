from .ActiveLearner.BaseActiveLearner import BaseActiveLearner
from .DataSetParsers.DataSetBuilder import DataSetBuilder
from .Trainable.BaseTrainableBuilder import BaseTrainableBuilder
from .Trainer.BaseTrainer import BaseTrainer
from .Dtos.DataSet import DataSet
from .Dtos.Enums.DataSetType import DataSetType
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
        DataSetType[config.getSetting('DataSetType')],
        dataSet=dataSet,
        config=config
    )

    return trainableBuilder.build()

def main() -> int:
    config: Config = _getConfig()
    dataSet: Type[DataSet] = DataSetBuilder.build(config)
    dataSetType: DataSetType = DataSetType[config.getSetting('DataSetType')]

    activeLearner: Type[BaseActiveLearner] = ObjectFactory.build(
        BaseActiveLearner,
        dataSetType=dataSetType,
        config=config
    )

    iterResults: Type[IterationResults] = None
    while activeLearner.hasUpdate(dataSet, iterResults):
        dataSet = activeLearner.getUpdate(dataSet, iterResults)

        trainable: Type[Trainable] = _getTrainable(dataSet, config)
        trainer: Type[BaseTrainer] = ObjectFactory.build(
            BaseTrainer,
            dataSetType=dataSetType,
            trainable=trainable,
            config=config
        )

        trainer.train()

        iterResults = trainable.getIterationResults()

    return 0

if __name__ == '__main__':
    sys.exit(main())


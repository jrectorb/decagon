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
        config=config
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
    while activeLearner.hasUpdate(dataSet, iterResults):
        dataSet = activeLearner.getUpdate(dataSet, iterResults)

        trainable: Type[Trainable] = _getTrainable(dataSet, config)
        trainer: Type[BaseTrainer] = _getTrainer(dataSet.id, trainable, config)

        trainer.train()

        iterResults = trainable.getIterationResults()

    return 0

if __name__ == '__main__':
    sys.exit(main())


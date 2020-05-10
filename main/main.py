from ActiveLearner.BaseActiveLearner import BaseActiveLearner
from DataSet.DataSetBuilder import DataSetBuilder
from Trainable.BaseTrainable import BaseTrainable
from Trainer.BaseTrainer import BaseTrainer
from Dtos.DataSet import DataSet
from Dtos.IterationResults import IterationResults
from Utils.ArgParser import ArgParser
from Utils.Config import Config
from Utils.ObjectFactory import ObjectFactory

from typing import Type
import sys

def _getConfig() -> Config:
    args = ArgParser().Parse()
    return Config(args)

def main() -> int:
    config: Config = _getConfig()
    dataSet: Type[DataSet] = DataSetBuilder.build(config)

    activeLearner: Type[BaseActiveLearner] = ObjectFactory.build(
        BaseActiveLearner,
        config
    )

    iterResults: Type[IterationResults] = None
    while activeLearner.hasUpdate(dataSet, iterResults):
        dataSet = activeLearner.getUpdate(dataSet, iterResults)

        trainable: Type[BaseTrainable] = ObjectFactory.build(BaseTrainable, dataSet, config)
        trainer: Type[BaseTrainer] = ObjectFactory.build(BaseTrainer, trainable, config)

        trainer.train()

        iterResults = trainable.getIterationResults()

    return 0

if __name__ == '__main__':
    sys.exit(main())


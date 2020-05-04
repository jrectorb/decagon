from typing import Type, Callable

from .AdjacencyMatricesTypes import AdjacencyMatricesTypes
from ..Utils.Config import Config

AdjMtxBuilder = Type[AdjacencyMatricesBuilder]

class AdjacencyMatricesBuilderFactory:
    @staticmethod
    def buildBuilder(baseCls, **kwargs):
        initializer = BaseAdjacencyMatricesBuilder.initializers[baseCls][dataSetType]

        return initializer(**kwargs)


from typing import Type, Callable

from .AdjacencyMatricesTypes import AdjacencyMatricesTypes
from ..Utils.Config import Config

AdjMtxBuilder = Type[AdjacencyMatricesBuilder]

class AdjacencyMatricesBuilderFactory:
    @staticmethod
    def buildBuilder(config: Config) -> AdjMtxBuilder:
        adjacencyMatricesType: AdjacencyMatricesType = config.getSetting(
            'AdjacencyMatricesType'
        )

        initializer: Callable[[Config], AdjMtxBuilder] = \
            BaseAdjacencyMatricesBuilder.initializers[adjacencyMatricesType]

        return initializer(config)


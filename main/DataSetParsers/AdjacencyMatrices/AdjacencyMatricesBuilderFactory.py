from typing import Type, Callable

from .AdjacencyMatricesTypes import AdjacencyMatricesTypes
from ..Utils.Config import Config

AdjMtxBuilder = Type[AdjacencyMatricesBuilder]

class AdjacencyMatricesBuilderFactory:
    @staticmethod
    def buildBuilder(
        nodeLists: NodeLists,
        dataSetType: DataSetType,
        config: Config
    ) -> AdjMtxBuilder:
        initializer: Callable[[Config], AdjMtxBuilder] = \
            BaseAdjacencyMatricesBuilder.initializers[dataSetType]

        return initializer(nodeLists, config)


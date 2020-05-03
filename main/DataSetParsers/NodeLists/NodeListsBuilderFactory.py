from typing import Type, Callable

from .AdjacencyMatricesTypes import AdjacencyMatricesTypes
from ..Utils.Config import Config

AdjMtxBuilder = Type[NodeListsBuilder]

class NodeListsBuilderFactory:
    @staticmethod
    def buildBuilder(nodeLists: NodeLists, config: Config) -> AdjMtxBuilder:
        dataSetType: DataSetType = config.getSetting(
            'DataSetType'
        )

        initializer: Callable[[Config], AdjMtxBuilder] = \
            BaseNodeListsBuilder.initializers[adjacencyMatricesType]

        return initializer(nodeLists, config)


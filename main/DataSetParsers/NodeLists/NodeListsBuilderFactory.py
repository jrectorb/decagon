from typing import Type, Callable

from .AdjacencyMatricesTypes import AdjacencyMatricesTypes
from ..Utils.Config import Config

NodeListsBuilder = Type[BaseNodeListsBuilder]

class NodeListsBuilderFactory:
    @staticmethod
    def buildBuilder(
        nodeLists: NodeLists,
        dataSetType: DataSetType,
        config: Config
    ) -> AdjMtxBuilder:
        initializer: Callable[[Config], NodeListsBuilder] = \
            BaseNodeListsBuilder.initializers[dataSetType]

        return initializer(nodeLists, config)


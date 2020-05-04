from typing import Type, Callable

from .AdjacencyMatricesTypes import AdjacencyMatricesTypes
from ..Utils.Config import Config

FeaturesBuilder = Type[BaseNodeFeaturesBuilder]

class NodeFeaturesBuilderFactory:
    @staticmethod
    def buildBuilder(dataSetType: DataSetType, config: Config) -> FeaturesBuilder:
        initializer: Callable[[Config], FeaturesBuilder] = \
            BaseNodeFeaturesBuilder.initializers[dataSetType]

        return initializer(config)


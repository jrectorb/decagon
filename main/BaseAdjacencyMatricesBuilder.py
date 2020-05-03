from abc import ABCMeta, abstractmethod
from typings import Map

Initializer = Callable[[Config], None]

class BaseAdjacencyMatricesBuilder(metaclass=ABCMeta):
    initializers: ClassVar[Dict[AdjacencyMatricesType, Initializer]] = {}

    def __init_subclass__(
        cls,
        adjacencyMatricesType: AdjacencyMatricesType,
        **kwargs
    ) -> None:
        super().__init_subclass__(cls, **kwargs)
        initializers[adjacencyMatricesType] = cls.__init__

    @abstractmethod
    def __init__(self, config: Config) -> None:
        pass

    @abstractmethod
    def build() -> AdjacencyMatrices:
        pass


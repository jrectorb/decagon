from abc import ABCMeta, abstractmethod
from typings import Dict, TypeVar

Initializer = Callable[[Config], None]

class BaseDataSetBuilder(metaclass=ABCMeta):
    initializers: ClassVar[Dict[DataSetType, Initializer]] = {}

    def __init_subclass__(
        cls,
        dataSetType: DataSetType,
        **kwargs
    ) -> None:
        super().__init_subclass__(cls, **kwargs)
        initializers[dataSetType] = cls.__init__

    @abstractmethod
    def __init__(self, config: Config) -> None:
        pass

    @abstractmethod
    def build() -> NodeFeatures:
        pass


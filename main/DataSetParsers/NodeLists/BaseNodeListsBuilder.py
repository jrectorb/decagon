from abc import ABCMeta, abstractmethod
from typings import Map

Initializer = Callable[[Config], None]

class BaseNodeListsBuilder(metaclass=ABCMeta):
    initializers: ClassVar[Dict[DataSetType, Initializer]] = {}

    def __init_subclass__(
        cls,
        dataSetType: DataSetType,
        **kwargs
    ) -> None:
        super().__init_subclass__(cls, **kwargs)
        initializers[DataSetType] = cls.__init__

    @abstractmethod
    def __init__(self, config: Config) -> None:
        pass

    @abstractmethod
    def build(self) -> NodeLists:
        pass


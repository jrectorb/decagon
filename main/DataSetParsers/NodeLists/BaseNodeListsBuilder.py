from abc import ABCMeta, abstractmethod

class BaseNodeListsBuilder(BaseFactorizableClass, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config: Config) -> None:
        pass

    @abstractmethod
    def build(self) -> NodeLists:
        pass


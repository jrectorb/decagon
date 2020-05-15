from .BaseFactorizableClass import BaseFactorizableClass
from enum import Enum
from typing import Type

class ObjectFactory:
    @staticmethod
    def build(baseCls, functionalityType: Type[Enum], **kwargs):
        cls = BaseFactorizableClass.classes[baseCls][functionalityType]
        return cls(**kwargs)


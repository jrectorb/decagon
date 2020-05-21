from abc import ABCMeta
from enum import Enum
from typing import ClassVar, Dict, Callable, Type
from .Config import Config
import inspect

ClassesDictType = ClassVar[Dict[type, Dict[Type[Enum], type]]]

class BaseFactorizableClass(metaclass=ABCMeta):
    '''
    This class is a base class to provide for other base
    classes which may have factories operate on top of them.
    Thus, in this context, Factorizable refers to the software
    pattern of object factories instead of something else, e.g.,
    matrix factorization.
    '''

    classes: ClassesDictType = {}

    def __init_subclass__(cls, functionalityType: Type[Enum] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        print(cls)

        if functionalityType is not None:
            lookupClass = cls._getBaseType()
            if lookupClass not in BaseFactorizableClass.classes:
                BaseFactorizableClass.classes[lookupClass] = {}

            BaseFactorizableClass.classes[lookupClass][functionalityType] = cls

    @classmethod
    def _getBaseType(cls) -> type:
        clsAncestors = set(inspect.getmro(cls))
        thisClassDirectDescendants = set(BaseFactorizableClass.__subclasses__())

        intersection = clsAncestors & thisClassDirectDescendants

        if len(intersection) != 1:
            raise TypeError(
                'Base type intersection must have exactly one class, ' +
                'but had %d classes instead' % len(intersection)
            )

        return intersection.pop()


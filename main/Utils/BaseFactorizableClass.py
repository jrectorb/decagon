from abc import ABCMeta
from typing import ClassVar, Dict, Callable
from .Config import Config
from ..Dtos.Enums.DataSetType import DataSetType
import inspect

InitializersDictType = ClassVar[Dict[type, Dict[DataSetType, Callable]]]

class BaseFactorizableClass(metaclass=ABCMeta):
    '''
    This class is a base class to provide for other base
    classes which may have factories operate on top of them.
    Thus, in this context, Factorizable refers to the software
    pattern of object factories instead of something else, e.g.,
    matrix factorization.
    '''

    initializers: InitializersDictType = {}

    def __init_subclass__(cls, dataSetType: DataSetType, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        print(cls)

        if dataSetType is not None:
            lookupClass = cls._getBaseType()
            if lookupClass not in BaseFactorizableClass.initializers:
                BaseFactorizableClass.initializers[lookupClass] = {}

            BaseFactorizableClass.initializers[lookupClass][dataSetType] = cls.__init__

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


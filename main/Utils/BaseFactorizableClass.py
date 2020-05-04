from abc import ABCMeta
from typings import ClassVar, Dict
from .Config import Config
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

    def __init__subclass__(cls, dataSetType: DataSetType, **kwargs) -> None:
        super().__init_subclass__(cls, **kwargs)

        if dataSetType is not None:
            lookupClass = cls._getBaseType()
            if lookupClass not in initializers:
                initializers[lookupClass] = {}

            initializers[lookupClass][dataSetType] = cls.__init__

    @classmethod
    def _getBaseType(cls) -> type:
        clsAncestors = set(inspect.getmro(cls))
        thisClassDirectDescendants = set(BaseFactorizableClass.__subclasses__)

        intersection = clsAncestors & thisClassDirectDescendants

        if len(intersection) != 1:
            raise TypeError(
                'Base type intersection must have exactly one class, ' +
                'but had %d classes instead' % len(intersection)
            )

        return intersection.pop()


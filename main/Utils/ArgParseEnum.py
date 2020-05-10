from enum import Enum, EnumMeta

#class ArgParseEnumMeta(EnumMeta):
#    def __call__(cls, value, *args, **kwargs):
#        if isinstance(value, str) and hasattr(cls, 'fromString'):
#
#            value = cls.__dict__[value]
#
#        super().__call__(cls, value, *args, **kwargs)


class ArgParseEnum(Enum):#, metaclass=ArgParseEnumMeta):
    """
    You may inherit from this enum to use a setting compatible with
    argparse.
    """
    def __new__(cls, value):
        if isinstance(value, str):
            value = cls.__dict__[value]

        return super().__new__(cls, value)

    def __str__(self):
        return self.name

    @classmethod
    def fromString(cls, val):
        try:
            return cls.__dict__[val]
        except:
            raise ValueError()


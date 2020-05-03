from enum import Enum

class RelationEnum(Enum):
    """
    You may inherit from this enum to use a setting compatible with
    argparse.
    """

    def __str__(self):
        return self.name

    @staticmethod
    def fromString(val):
        try:
            return self.__class__[val]
        except:
            raise ValueError()


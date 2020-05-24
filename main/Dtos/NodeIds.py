from typing import Type

class BaseNodeId(int):
    '''
    Not meant to be instantiated, but rather used as a utility for
    other Id classes (namely, to convert from decagon format)
    '''
    def __new__(cls, val: object) -> Type['BaseNodeId']:
        if isinstance(val, str):
            val = BaseNodeId._formatStr(val)

        return int(val)

    @classmethod
    def fromDecagonFormat(cls, val: object) -> Type['BaseNodeId']:
        return cls(BaseNodeId._formatStr(val))

    @staticmethod
    def _formatStr(preStr: str) -> str:
        '''
        Removes letters and prefixed 0s from a string  E.g.,

        CID0012314 -> 12314
        C00512341  -> 512341
        SID123     -> 123

        '''
        # Remove leters from string
        preStr = ''.join(filter(str.isdigit, preStr))

        # Remove prefixed 0s
        preStr = preStr.lstrip('0')

        return preStr

class DrugId(BaseNodeId):
    pass

class ProteinId(BaseNodeId):
    pass

class SideEffectId(BaseNodeId):
    pass


from .BaseFactorizableClass import BaseFactorizableClass

class ObjectFactory:
    @staticmethod
    def build(baseCls, dataSetType, **kwargs):
        cls = BaseFactorizableClass.classes[baseCls][dataSetType]
        return cls(**kwargs)


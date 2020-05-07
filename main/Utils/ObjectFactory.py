from .BaseFactorizableClass import BaseFactorizableClass

class ObjectFactory:
    @staticmethod
    def buildObject(baseCls, **kwargs):
        initializer = BaseFactorizableClass.initializers[baseCls][dataSetType]

        return initializer(**kwargs)


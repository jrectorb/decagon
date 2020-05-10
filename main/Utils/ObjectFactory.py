from .BaseFactorizableClass import BaseFactorizableClass

class ObjectFactory:
    @staticmethod
    def build(baseCls, **kwargs):
        initializer = BaseFactorizableClass.initializers[baseCls][dataSetType]

        return initializer(**kwargs)


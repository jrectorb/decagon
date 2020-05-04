class ObjectFactory:
    @staticmethod
    def buildObject(baseCls, **kwargs):
        initializer = BaseAdjacencyMatricesBuilder.initializers[baseCls][dataSetType]

        return initializer(**kwargs)


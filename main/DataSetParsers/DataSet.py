class DataSet:
    def __init__(
        self,
        adjacencyMatrices: AdjacencyMatrices,
        nodeFeatures: NodeFeatures
    ) -> None:
        self.adjacencyMatrices: AdjacencyMatrices = adjacencyMatrices
        self.nodeFeatures: NodeFeatures = nodeFeatures


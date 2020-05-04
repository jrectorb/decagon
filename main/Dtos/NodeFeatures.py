import scipy.sparse as sp

class NodeFeatures:
    def __init__(
        self,
        proteinNodeFeatures: sp.coo_matrix,
        drugNodeFeatures: sp.coo_matrix
    ) -> None:
        self.proteinNodeFeatures: sp.coo_matrix = proteinNodeFeatures
        self.drugNodeFeatures: sp.coo_matrix = drugNodeFeatures


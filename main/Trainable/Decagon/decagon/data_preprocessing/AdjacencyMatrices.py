class AdjacencyMatrices:
    def __init__(
        self,
        drugDrugRelationMtxs,
        drugProteinRelationMtx,
        ppiMtx
    ):
        # drugDrugRelationMtxs is a dictionary where each (key, value) pair
        # has the key as a string representation of the STITCH side effect
        # ID and value as the adjacency matrices for that STITCH side
        # effect ID
        self.drugDrugRelationMtxs = drugDrugRelationMtxs

        # drugProteinRelationMtx is the adjacency matrix for relations between
        # the drug and proteins
        self.drugProteinRelationMtx = drugProteinRelationMtx

        # ppiMtx is an adjacency matrix for interactions between proteins
        self.ppiMtx = ppiMtx





RelationIDToGraph = Dict[str, Type[nx.Graph]]
RelationIDToSparseMtx = Dict[str, Type[sp.spmatrix]]

class DecagonPublicDataAdjacencyMatricesBuilder(
    BaseAdjacencyMatricesBuilder,
    adjacencyMatricesType = AdjacencyMatricesType.DecagonDummyData
):
    def __init__(self, nodeLists: NodeLists, config: Config) -> None:
        self.numDrugDrugRelationTypes: int = config.getInt('NumDrugDrugRelationTypes')
        self.numProteins: int              = len(nodeLists.proteinNodes)
        self.numDrugs: int                 = len(nodeLists.drugNodes)

    def build(self) -> AdjacencyMatrices:
        drugProteinRelationMtx: sp.csr_matrix = self._buildDrugProteinRelationMtx()
        drugDrugRelationMtxs: RelationIDToSparseMtx = \
            self._buildDrugDrugRelationMtxs(drugProteinRelationMtx)

        return AdjacencyMatrices(
            drugDrugRelationMtxs=self._buildDrugDrugRelationMtxs(),
            drugProteinRelationMtx=self._buildDrugProteinRelationMtx(),
            ppiMtx=self._buildPpiMtx(),
        )

    def _buildDrugProteinRelationMtx(self) -> sp.csr_matrix:
        preMtx = 10 * .random.randn(self.numProteins, self.numDrugs)
        binaryMtx = (preMtx > 15).astype(int)

        return sp.csr_matrix(binaryMtx)

    def _buildDrugDrugRelationMtxs(
        self,
        drugProteinMtx: sp.csr_matrix
    ) -> RelationIDToSparseMtx:
        result: RelationIDToSparseMtx = {}

        tmp = np.dot(drugProteinMtx, drugProteinMtx.transpose(copy=True))
        for i in range(self.numDrugDrugRelationTypes):
            mat = np.zeros((self.numDrugs, self.numDrugs))

            for d1, d2 in combinations(list(range(self.numDrugs)), 2):
                if tmp[d1, d2] == i + 4:
                    mat[d1, d2] = mat[d2, d1] = 1.

            result[str(i)] = mat

        return result

    def _buildPpiMtx(self) -> sp.csr_matrix:
        plantedGraph = nx.planted_partition_graph(50, 10, 0.2, 0.05, seed=42)
        return nx.adjacency_matrix(plantedGraph)


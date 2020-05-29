from threading import Lock

def _getConfig() -> Config:
    argParser = ArgParser()
    argParser.parse()

    return Config(argParser)

config = _getConfig()

# Will be set in the NpPredictor class
predsInfoHolder = None
predsInfoHolderLock = Lock()

# Internal class
# This should only be instantiated once
class _PredictionsInfoHolder:
    def __init__(self):
        self.nodeLists: NodeLists = self._getDrugNodeList()
        self.drugIdToIdx = {
            drugId: idx
            for idx, drugId in enumerate(self.nodeLists.drugNodeList)
        }

        npSaveDir = config.getSetting('NpSaveDir')

        embeddingsFname = npSaveDir + 'embeddings.npy'
        self.embeddings = np.load(embeddingsFile)

        globRelFname = npSaveDir + 'GlobalRelations.npy'
        self.globalInteraction = np.load(globRelFname)

        self.testEdgeDict = self._buildTestEdgeDict()
        self.trainEdgeDict = self._buildTrainEdgeDict()

    def _getDrugNodeLists(self) -> NodeLists:
        listBuilder = ObjectFactory.build(
            BaseNodeListsBuilder,
            DataSetType[config.getSetting('DataSetType')],
            config=config,
        )

        return listBuilder.build()

    def _buildTestEdgeDict(self) -> Dict:
        result = {}

        testEdgeReader = self._getTestEdgeReader()
        for row in testEdgeReader:
            if not self._isRowValid(row):
                continue

            relId = row['RelationId']
            if relId not in result:
                result[relId] = np.array()

            fromNodeIdx = self.drugIdToIdx[row['FromNode']]
            toNodeIdx = self.drugIdToIdx[row['ToNode']]
            newArr = np.array([fromNodeIdx, toNodeIdx, row['Label']])

            result[relId] = np.append(result[relId], newArr)

        return result

    def _isRowValid(self, row):
        def _isDrugNode(strVal: str) -> bool:
            return strVal[:3] == 'CID'

        return _isDrugNode(row['FromNode']) and _isDrugNode(row['ToNode'])

    def _buildTrainEdgeDict(self) -> None:
        result = {}

        # Define indices here to not redefine it a bunch
        indices = None
        for relId, mtx in self._getDrugDrugMtxs():
            if indices is None:
                indices = self._getIndices(mtx.shape)

            trainEdgeIdxs = self._getTrainEdgeIdxs(indices, relId, mtx.shape)
            trainEdgeLabels = self._getTrainEdgeLabels(mtx, trainEdgeIdxs)

            result[relId] = np.hstack([trainEdgeIdxs, trainEdgeLabels])

        return result

    def _getIndices(self, shape) -> np.ndarray:
        xx, yy = np.indices(shape)
        return np.dstack([xx, yy]).reshape((-1, 2))

    def _getTrainEdgeIdxs(
        self,
        indices: np.ndarray,
        relId: str,
        mtxShape: Tuple[int, int]
    ) -> np.ndarray:
        # Get test edges and remove that labels from indices (slice of :2)
        testEdges = self.testEdgeDict[relId][:, :2]

        indicesLinear   = (indices[:, 0] * mtxShape[1]) + indices[:, 1]
        testEdgesLinear = (testEdges[:, 0] * mtxShape[1]) + testEdges[:, 1]

        trainEdges = np.setdiff1d(indicesLinear, testEdgesLinear)

        return np.dstack(np.unravel_index(trainEdges, mtxShape)).rreshape(-1, 2)

    def _getTrainEdgeLabels(self, mtx: np.ndarray, edgeIdxs: np.ndarray) -> np.ndarray:
        idxsLinear = (edgeIdxs[:, 0] * mtx.shape[1]) + edges[:, 1]
        return mtx[idxsLinear]

    def _getDrugDrugMtxs(self):
        adjMtxBuilder = ObjectFactory.build(
            BaseAdjacencyMatricesBuilder,
            DataSetType[config.getSetting('DataSetType')],
            config=config,
            nodeLists=self.nodeLists
        )

        return adjMtxBuilder.build().drugDrugRelationMtxs

    def _getTestEdgeReader(self) -> csv.DictReader:
        testEdgeFname = config.getSetting('TestEdgeFilename')
        return csv.DictReader(open(testEdgeFname))

class TrainingEdgeIterator:
    def __init__(self, relationId: str) -> None:
        self._initGlobalInfosHolderIfNeeded()

        self.relationId = relationId

    def _initGlobalInfosHolderIfNeeded(self) -> None:
        if predsInfoHolder is None:
            predsInfoHolderLock.acquire()
            if predsInfoHolder is None:
                predsInfoHolder = _PredictionsInfoHolder()

            predsInfoHolderLock.release()

    # Returns 3-dim ndarray where the first column is the from node,
    # the second column is the to node, and the third column is the edge label
    def get_train_edges(self) -> np.ndarray:
        return predsInfoHolder.trainEdges[self.relationId]


class NpPredictor:
    def __init__(self, relationId: str) -> None:
        self._initGlobalInfosHolderIfNeeded()

        npSaveDir = config.getSetting('NpSaveDir')
        relFname = 'EmbeddingImportance-%s.npy' % relationId
        self.defaultImportanceMtx = np.load(relFname)

        baseTestEdges = predsInfoHolder.testEdgeDict[relationId]
        self.negTestEdges = baseTestEdges[baseTestEdges[:, 2] == 0]
        self.posTestEdges = baseTestEdges[baseTestEdges[:, 2] == 1]

    def _initGlobalInfosHolderIfNeeded(self) -> None:
        if predsInfoHolder is None:
            predsInfoHolderLock.acquire()
            if predsInfoHolder is None:
                predsInfoHolder = _PredictionsInfoHolder()

            predsInfoHolderLock.release()

    def predict(self, importance_matrix=None):
        importanceMtx = self.defaultImportanceMtx
        if importance_matrix is not None:
            importanceMtx = importance_mtx

        rowEmbeddings = self._getRowEmbeddings()
        colEmbeddings = self._getColEmbeddings()
        globalInteractionMtx = self._getGlobalInteractionMtx

        rawPreds = colEmbeddings @ importanceMtx @ globalInteractionMtx @ importanceMtx @ rowEmbeddings

        # This will process to something like a 4-dim data frame with
        # the relevant information for each prediction (i.e., embeddings,
        # global interaction mtx, importance matrix)
        return self._toDataFrame(rawPreds)



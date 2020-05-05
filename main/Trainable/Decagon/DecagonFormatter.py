from typing import Dict, List
import tensorflow as tf
import scipy.sparse as sp

InteractionSubGraphType = int
EdgeType                = tuple
StrDecoderSpecifier     = str

EdgeTypeMatrixDimensionsDict = Dict[EdgeType, List[tuple]]
EdgeTypeAdjacencyMatrixDict  = Dict[EdgeType, List[sp.coo_matrix]]
EdgeTypeDecoderDict          = Dict[EdgeType, StrDecoderSpecifier]
EdgeTypeNumMatricesDict      = Dict[EdgeType, int]
FeaturesDict                 = Dict[InteractionSubGraphType, sp.coo_matrix]
DegreesDict                  = Dict[InteractionSubGraphType, List[int]]
PlaceholdersDict             = Dict[str, tf.placeholder]
Flags                        = tf.python.platform.flags._FlagValuesWrapper

class DecagonDataSet:
    PPI_GRAPH_IDX = 0
    DRUG_DRUG_GRAPH_IDX = 1

    PPI_GRAPH_EDGE_TYPE   = (PPI_GRAPH_IDX, PPI_GRAPH_IDX)
    DRUG_DRUG_EDGE_TYPE   = (DRUG_DRUG_GRAPH_IDX, DRUG_DRUG_GRAPH_IDX)
    PPI_TO_DRUG_EDGE_TYPE = (PPI_GRAPH_IDX, DRUG_DRUG_GRAPH_IDX)
    DRUG_TO_PPI_EDGE_TYPE = (DRUG_DRUG_GRAPH_IDX, PPI_GRAPH_IDX)

    '''
    This class contains all the necessary information to correctly
    instantiate a decagon batch iterator, optimizer, and model.  It
    also contains methods to build a DecagonDataSet from a base
    DataSet object.
    '''
    def __init__(
        self,
        adjacencyMatrixDict: EdgeTypeAdjacencyMatrixDict,
        edgeTypeMatrixDimDict: EdgeTypeMatrixDimensionsDict,
        edgeTypeNumMatricesDict: EdgeTypeNumMatricesDict,
        featuresDict: FeaturesDict,
        degreesDict: DegreesDict,
        config: Config
    ) -> None:
        self.adjacencyMatrixDict: EdgeTypeAdjacencyMatrixDict = adjacencyMatrixDict
        self.edgeTypeMatrixDimDict: EdgeTypeMatrixDimensionsDict = edgeTypeMatrixDimDict
        self.edgeTypeNumMatricesDict: EdgeTypeNumMatricesDict = edgeTypeNumMatricesDict
        self.featuresDict: FeaturesDict = featuresDict
        self.degreesDict: DegreesDict = degreesDict

        self.edgeTypeDecoderDict: EdgeTypeDecoderDict = self._getEdgeTypeDecoderDict(config)
        self.placeholdersDict: PlaceholdersDict = self._getPlaceholdersDict(config)
        self.flags: Flags = self._getFlags(config)

    def _getEdgeTypeDecoderDict(self, config: Config) -> EdgeTypeDecoderDict:
        pass

    def _getPlaceholdersDict(self, config: Config) -> PlaceholderDict:
        pass

    def _getFlags(self, config: Config) -> Flags:
        pass

    @staticmethod
    def fromDataSet(dataSet: DataSet, config: Config) -> DecagonDataSet:
        adjMtxDict = DecagonDataSet._getAdjMtxDict(
            dataSet.adjacencyMatrices,
            config
        )

        featuresDict = DecagonDataSet._getFeaturesDict(dataSet.nodeFeatures)

        edgeTypeMatrixDimDict = DecagonDataSet._getEdgeTypeMtxDimDict(adjMtxDict)
        edgeTypeNumMatricesDict = DecagonDataSet._getEdgeTypeNumMatricesDict(adjMtxDict)
        degreesDict = DecagonDataSet._getDegreesDict(adjMtxDict)

        return DecagonDataSet(
            adjMtxDict,
            edgeTypeMatrixDimDict,
            edgeTypeNumMatricesDict,
            featuresDict,
            degreesDict,
            config,
        )

    @staticmethod
    def _getAdjMtxDict(
        adjMtxs: AdjacencyMatrices,
        config: Config
    ) -> EdgeTypeAdjacencyMatrixDict:
        result: EdgeTypeAdjacencyMatrixDict = defaultdict(list)

        result[DecagonDataSet.PPI_GRAPH_EDGE_TYPE] = [adjMtxs.proteinProteinRelationMtx]
        result[DecagonDataSet.PPI_TO_DRUG_EDGE_TYPE] = [adjMtxs.drugProteinRelationMtx]
        result[DecagonDataSet.DRUG_DRUG_EDGE_TYPE] = adjMtxs.drugDrugRelationMtxs

        # Decagon's original code uses transposed matrices to train as well
        # as original matrices.  Here we provide the option to do so too.
        useTransposedMtxs = bool(
            config.getSetting('TrainWithTransposedAdjacencyMatrices')
        )
        if useTransposedMtxs == True:
            DecagonDataSet._augmentAdjMtxDictWithTranspose(result)

        return result

    @staticmethod
    def _augmentAdjMtxDictWithTranspose(
        adjMtxDict: EdgeTypeAdjacencyMatrixDict
    ) -> None:
        for edgeType, mtxs in adjMtxDict.items():
            resEdgeType = None
            if edgeType == DecagonDataSet.PPI_TO_DRUG_EDGE_TYPE
                resEdgeType = DecagonDataSet.DRUG_TO_PPI_EDGE_TYPE
            else:
                resEdgeType = edgeType

            adjMtxDict[resEdgeType].extend([
                mtx.transpose(copy=True) for mtx in adjMtxDict[edgeType]
            ])

        return

    @staticmethod
    def _getFeaturesDict(nodeFeatures: NodeFeatures):
        return {
            DecagonDataSet.PPI_GRAPH_IDX: nodeFeatures.proteinNodeFeatures,
            DecagonDataSet.DRUG_DRUG_GRAPH_IDX: nodeFeatures.drugNodeFeatures,
        }

    @staticmethod
    def _getEdgeTypeMtxDimDict(
        adjMtxDict: EdgeTypeAdjacencyMatrixDict
    ) -> EdgeTypeMatrixDimensionsDict:
        return {
            edgeType: [mtx.shape for mtx in mtxs]
            for edgeType, mtxs in adjMtxDict.items()
        }

    @staticmethod
    def _getEdgeTypeNumMatricesDict(
        adjMtxDict: EdgeTypeAdjacencyMatrixDict
    ) -> EdgeTypeNumMatricesDict:
        return { edgeType: len(mtxs) for edgeType, mtxs in adjMtxDict.items() }

    @staticmethod
    def _getDegreesDict(
        adjMtxDict: EdgeTypeAdjacencyMatrixDict
    ) -> DegreesDict:
        def getDegrees(mtx: sp.coo_matrix) -> int:
            return np.array(mtx.sum(axis=0)).squeeze()

        def getDegreesList(mtxs: List[sp.coo_matrix]) -> List[int]:
            return [getDegrees(mtx) for mtx in mtxs]

        ppiMtxs = adjMtxDict[DecagonDataSet.PPI_GRAPH_EDGE_TYPE]
        drugDrugMtxs = adjMtxDict[DecagonDataSet.DRUG_DRUG_GRAPH_EDGE_TYPE]

        return {
            PPI_GRAPH_IDX: getDegreesList(ppiMtxs),
            DRUG_DRUG_GRAPH_IDX: getDegreesList(drugDrugMtxs),
        }


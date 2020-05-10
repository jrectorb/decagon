from ...Utils.Config import Config
from ...Dtos.AdjacencyMatrices import AdjacencyMatrices
from ...Dtos.NodeFeatures import NodeFeatures
from ...Dtos.DataSet import DataSet
from ...Dtos.TypeShortcuts import PlaceholdersDict
from tensorflow.python.platform import flags as tfFlags
from typing import Dict, List, Type
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
Flags                        = tfFlags._FlagValuesWrapper

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
        validDecoders = set(['innerproduct', 'distmult', 'bilinear', 'dedicom'])

        result: EdgeTypeDecoderDict = {}

        result[PPI_GRAPH_EDGE_TYPE] = config.getSetting('PPIEdgeDecoder')
        result[PPI_TO_DRUG_EDGE_TYPE] = config.getSetting('ProteinToDrugEdgeDecoder')
        result[DRUG_DRUG_EDGE_TYPE] = config.getSetting('DrugDrugEdgeDecoder')

        if DecagonDataSet._shouldTranspose(config):
            result[DRUG_TO_PPI_EDGE_TYPE] = config.getSetting('DrugProteinEdgeDecoder')

        return result

    def _getPlaceholdersDict(
        self,
        edgeTypeNumMatricesDict: EdgeTypeNumMatricesDict
    ) -> PlaceholdersDict:
        result: PlaceholdersDict = {}

        result['batch'] = tf.placeholder(tf.int32, name='batch')
        result['degrees'] = tf.placeholder(tf.int32)
        result['dropout'] = tf.placeholder_with_default(0., shape=())

        result['batch_edge_type_idx'] = tf.placeholder(
            tf.int32,
            shape=(),
            name='batch_edge_type_idx'
        )

        result['batch_row_edge_type'] = tf.placeholder(
            tf.int32,
            shape=(),
            name='batch_row_edge_type'
        )

        result['batch_col_edge_type'] = tf.placeholder(
            tf.int32,
            shape=(),
            name='batch_col_edge_type'
        )

        for edgeType, numMtxsForEdgeType in edgeTypeNumMatricesDict.item():
            for i in range(numMtxsForEdgeType):
                key = 'adj_mats_%d,%d,%d' % (edgeType[0], edgeType[1], i)
                result[key] = tf.sparse_placeholder(tf.float32)

        for x in [DecagonDataSet.PPI_GRAPH_IDX, DecagonDataSet.DRUG_DRUG_GRAPH_IDX]:
            result['feat_%d' % x] = tf.sparse_placeholder(tf.float32)

        return placeholders

    def _getFlags(self, config: Config) -> Flags:
        flags = tf.app.flags

        def defVal(key: str, desc: str, typeToDef: type) -> None:
            defFxn = None
            if typeToDef == int:
                defFxn = flags.DEFINE_integer
            elif typeToDef == float:
                defFxn = flags.DEFINE_float
            elif typeToDef == bool:
                defFxn = flags.DEFINE_boolean
            else:
                raise TypeError('Invalid type')

            defFxn(key, typeToDef(config.getSetting(key)), desc)

        defVal('neg_sample_size', 'Negative sample size.', int)
        defVal('learning_rate', 'Initial learning rate.', float)
        defVal('epochs', 'Number of epochs to train.', int)
        defVal('hidden1', 'Number of units in hidden layer 1.', int)
        defVal('hidden2', 'Number of units in hidden layer 2.', int)
        defVal('weight_decay', 'Weight for L2 loss on embedding matrix.', float)
        defVal('dropout', 'Dropout rate (1 - keep probability).', float)
        defVal('max_margin', 'Max margin parameter in hinge loss', float)
        defVal('batch_size', 'Minibatch size.', int)
        defVal('bias', 'Bias term.', bool)

        return flags.FLAGS

    @staticmethod
    def fromDataSet(dataSet: DataSet, config: Config) -> Type['DecagonDataSet']:
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
        if DecagonDataSet._shouldTranspose(config):
            DecagonDataSet._augmentAdjMtxDictWithTranspose(result)

        return result

    @staticmethod
    def _shouldTranspose(config: Config) -> bool:
        strVal = config.getSetting('TrainWithTransposedAdjacencyMatrices')
        return bool(strVal)

    @staticmethod
    def _augmentAdjMtxDictWithTranspose(
        adjMtxDict: EdgeTypeAdjacencyMatrixDict
    ) -> None:
        for edgeType, mtxs in adjMtxDict.items():
            resEdgeType = None
            if edgeType == DecagonDataSet.PPI_TO_DRUG_EDGE_TYPE:
                resEdgeType = DecagonDataSet.DRUG_TO_PPI_EDGE_TYPE
            else:
                resEdgeType = edgeType

            adjMtxDict[resEdgeType].extend([
                mtx.transpose(copy=True) for mtx in adjMtxDict[edgeType]
            ])

        return

    @staticmethod
    def _getFeaturesDict(nodeFeatures: NodeFeatures) -> FeaturesDict:
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


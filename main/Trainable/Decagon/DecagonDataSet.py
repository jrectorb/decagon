from .decagon.utility import preprocessing
from ...Utils.Config import Config
from ...Dtos.AdjacencyMatrices import AdjacencyMatrices
from ...Dtos.NodeFeatures import NodeFeatures
from ...Dtos.DataSet import DataSet
from ...Dtos.TypeShortcuts import PlaceholdersDict
from collections import defaultdict
from tensorflow.python.platform import flags as tfFlags
from typing import Dict, List, Type, Iterable, Tuple
import tensorflow as tf
import scipy.sparse as sp
import numpy as np

InteractionSubGraphType = int
EdgeType                = tuple
StrDecoderSpecifier     = str

EdgeTypeMatrixDimensionsDict = Dict[EdgeType, List[tuple]]
EdgeTypeAdjacencyMatrixDict  = Dict[EdgeType, List[sp.coo_matrix]]
EdgeTypeDecoderDict          = Dict[EdgeType, StrDecoderSpecifier]
EdgeTypeNumMatricesDict      = Dict[EdgeType, int]
FeaturesTuple                = Tuple[np.ndarray, sp.coo_matrix, Tuple[int, int]]
FeaturesDict                 = Dict[InteractionSubGraphType, FeaturesTuple]
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

    HaveDefinedFlags = False

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
        print({e: [x.shape for x in y] for e,y in adjacencyMatrixDict.items()})
        self.adjacencyMatrixDict: EdgeTypeAdjacencyMatrixDict = adjacencyMatrixDict
        self.edgeTypeMatrixDimDict: EdgeTypeMatrixDimensionsDict = edgeTypeMatrixDimDict
        self.edgeTypeNumMatricesDict: EdgeTypeNumMatricesDict = edgeTypeNumMatricesDict
        self.featuresDict: FeaturesDict = featuresDict
        self.degreesDict: DegreesDict = degreesDict

        self.edgeTypeDecoderDict: EdgeTypeDecoderDict = self._getEdgeTypeDecoderDict(config)
        self.flags: Flags = self._getFlags(config)
        self.placeholdersDict: PlaceholdersDict = self._getPlaceholdersDict(
            edgeTypeNumMatricesDict
        )

    def _getEdgeTypeDecoderDict(self, config: Config) -> EdgeTypeDecoderDict:
        validDecoders = set(['innerproduct', 'distmult', 'bilinear', 'dedicom'])

        result: EdgeTypeDecoderDict = {}

        result[DecagonDataSet.PPI_GRAPH_EDGE_TYPE] = \
            config.getSetting('PPIEdgeDecoder')
        result[DecagonDataSet.PPI_TO_DRUG_EDGE_TYPE] = \
            config.getSetting('ProteinToDrugEdgeDecoder')
        result[DecagonDataSet.DRUG_DRUG_EDGE_TYPE] = \
            config.getSetting('DrugDrugEdgeDecoder')

        if DecagonDataSet._shouldTranspose(config):
            result[DecagonDataSet.DRUG_TO_PPI_EDGE_TYPE] = \
                config.getSetting('DrugProteinEdgeDecoder')

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

        for edgeType, numMtxsForEdgeType in edgeTypeNumMatricesDict.items():
            for i in range(numMtxsForEdgeType):
                key = 'adj_mats_%d,%d,%d' % (edgeType[0], edgeType[1], i)
                result[key] = tf.sparse_placeholder(tf.float32)

        for x in [DecagonDataSet.PPI_GRAPH_IDX, DecagonDataSet.DRUG_DRUG_GRAPH_IDX]:
            result['feat_%d' % x] = tf.sparse_placeholder(tf.float32)

        return result

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
            elif typeToDef == str:
                defFxn = flags.DEFINE_string
            else:
                raise TypeError('Invalid type')

            defFxn(key, typeToDef(config.getSetting(key)), desc)

        if not DecagonDataSet.HaveDefinedFlags:
            defVal('neg_sample_size', 'Negative sample size.', float)
            defVal('learning_rate', 'Initial learning rate.', float)
            defVal('epochs', 'Number of epochs to train.', int)
            defVal('hidden1', 'Number of units in hidden layer 1.', int)
            defVal('hidden2', 'Number of units in hidden layer 2.', int)
            defVal('weight_decay', 'Weight for L2 loss on embedding matrix.', float)
            defVal('dropout', 'Dropout rate (1 - keep probability).', float)
            defVal('max_margin', 'Max margin parameter in hinge loss', float)
            defVal('batch_size', 'Minibatch size.', int)
            defVal('bias', 'Bias term.', bool)

            # For compatibility with ray
            flags.DEFINE_string('node-ip-address', '', 'RayCompat')
            flags.DEFINE_string('node-manager-port', '', 'RayCompat')
            flags.DEFINE_string('object-store-name', '', 'RayCompat')
            flags.DEFINE_string('raylet-name', '', 'RayCompat')
            flags.DEFINE_string('redis-address', '', 'RayCompat')
            flags.DEFINE_string('config-list', '', 'RayCompat')
            flags.DEFINE_string('temp-dir', '', 'RayCompat')
            flags.DEFINE_string('redis-password', '', 'RayCompat')

            DecagonDataSet.HaveDefinedFlags = True

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
        result[DecagonDataSet.DRUG_DRUG_EDGE_TYPE] = list(adjMtxs.drugDrugRelationMtxs.values())

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
        tmp: EdgeTypeAdjacencyMatrixDict = {}
        for edgeType, mtxs in adjMtxDict.items():
            mtxs = DecagonDataSet._extractMtxs(adjMtxDict[edgeType])
            tMtxs = [mtx.transpose(copy=True, setId=True) for mtx in mtxs]

            if edgeType == DecagonDataSet.PPI_TO_DRUG_EDGE_TYPE:
                tmp[edgeType] = mtxs
                tmp[DecagonDataSet.DRUG_TO_PPI_EDGE_TYPE] = tMtxs

            else:
                tmp[edgeType] = mtxs + tMtxs

        for edgeType, mtxs in tmp.items():
            adjMtxDict[edgeType] = mtxs

        return

    # Do not type annotate the argument as it is either a dict or list
    @staticmethod
    def _extractMtxs(mtxContainer) -> List[sp.coo_matrix]:
        if isinstance(mtxContainer, list):
            return mtxContainer
        elif isinstance(mtxContainer, dict):
            return list(mtxContainer.values())
        else:
            raise TypeError('mtxContainer must be of type list or dict')

    @staticmethod
    def _getFeaturesDict(nodeFeatures: NodeFeatures) -> FeaturesDict:
        processedProteinFeatures = preprocessing.sparse_to_tuple(
            nodeFeatures.proteinNodeFeatures
        )

        processedDrugFeatures = preprocessing.sparse_to_tuple(
            nodeFeatures.drugNodeFeatures
        )

        return {
            DecagonDataSet.PPI_GRAPH_IDX: processedProteinFeatures,
            DecagonDataSet.DRUG_DRUG_GRAPH_IDX: processedDrugFeatures,
        }

    @staticmethod
    def _getEdgeTypeMtxDimDict(
        adjMtxDict: EdgeTypeAdjacencyMatrixDict
    ) -> EdgeTypeMatrixDimensionsDict:
        return {
            edgeType: [
                mtx.shape
                for mtx in DecagonDataSet._extractMtxs(adjMtxDict[edgeType])
            ]
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

        def getDegreesList(mtxs: Iterable[sp.coo_matrix]) -> List[int]:
            return [getDegrees(mtx) for mtx in mtxs]

        ppiMtxs = adjMtxDict[DecagonDataSet.PPI_GRAPH_EDGE_TYPE]
        drugDrugMtxsDict = adjMtxDict[DecagonDataSet.DRUG_DRUG_EDGE_TYPE]

        return {
            DecagonDataSet.PPI_GRAPH_IDX: getDegreesList(ppiMtxs),
            DecagonDataSet.DRUG_DRUG_GRAPH_IDX: getDegreesList(drugDrugMtxsDict),
        }


from .DecagonDataSet import DecagonDataSet
from .DecagonTrainable import DecagonTrainable
from ..BaseTrainableBuilder import BaseTrainableBuilder
from ...Dtos.DataSet import DataSet
from ...Dtos.Enums.TrainableType import TrainableType
from ...Dtos.IterationResults import IterationResults
from ...Dtos.NodeIds import DrugId, ProteinId, SideEffectId
from ...Dtos.Trainable import Trainable
from ...Utils.Config import Config
from .decagon.deep.minibatch import EdgeMinibatchIterator
from .decagon.deep.model import DecagonModel
from .decagon.deep.optimizer import DecagonOptimizer
from typing import Type, Dict, Tuple

import tensorflow as tf
import numpy as np
import datetime
import csv

DRUG_DRUG_GRAPH_TYPE = (1, 1)

FROM_GRAPH_IDX = 0
TO_GRAPH_IDX   = 1

FROM_NODE_IDX  = 0
TO_NODE_IDX    = 1

class DecagonTrainableBuilder(
    BaseTrainableBuilder,
    functionalityType=TrainableType.DecagonTrainable
):
    def __init__(
        self,
        dataSet: DataSet,
        drugDrugTestEdges: Dict[int, Dict[str, np.array]],
        config: Config,
        decagonDataSet: DecagonDataSet = None,
        placeholdersDict=None
    ) -> None:
        self.dataSet: DecagonDataSet = None
        if decagonDataSet is not None:
            self.dataSet = decagonDataSet
        else:
            self.dataSet = DecagonDataSet.fromDataSet(dataSet, config)

        drugDrugMtxs = dataSet.adjacencyMatrices.drugDrugRelationMtxs
        self.relIdxToRelId = {idx: relId for idx, relId in enumerate(drugDrugMtxs)}

        self.placeholdersDict = \
            placeholdersDict if placeholdersDict is not None else self.dataSet.placeholdersDict

        self.drugDrugTestEdges: Dict[int, Dict[str, np.array]] = drugDrugTestEdges
        self.nodeLists = dataSet.nodeLists
        self.config: Config = config

    def build(self) -> Type[Trainable]:
        dataSetIterator = self.getDataSetIterator()
        model = self.getModel()
        optimizer = self.getOptimizer(model)

        self._recordTestEdges(dataSetIterator)

        return DecagonTrainable(
            dataSetIterator,
            optimizer,
            model,
            self.dataSet.placeholdersDict
        )

    def getDataSetIterator(self) -> EdgeMinibatchIterator:
        return EdgeMinibatchIterator(
            self.dataSet.adjacencyMatrixDict,
            self.dataSet.featuresDict,
            self.dataSet.edgeTypeNumMatricesDict,
            self.drugDrugTestEdges,
            self.dataSet.flags.batch_size,
            float(self.config.getSetting('TestSetProportion'))
        )

    def getModel(self) -> DecagonModel:
        FEATURE_TPL_MTX_IDX   = 1
        FEATURE_TPL_SHAPE_IDX = 2

        subGraphToFeaturesDimDict = {
            subGraphIdx: featureTpl[FEATURE_TPL_SHAPE_IDX][1]
            for subGraphIdx, featureTpl in self.dataSet.featuresDict.items()
        }

        subGraphToNumNonZeroValsDict = {
            subGraphIdx: int(featureTpl[FEATURE_TPL_MTX_IDX].sum())
            for subGraphIdx, featureTpl in self.dataSet.featuresDict.items()
        }

        return DecagonModel(
            self.dataSet.placeholdersDict,
            subGraphToFeaturesDimDict,
            subGraphToNumNonZeroValsDict,
            self.dataSet.edgeTypeNumMatricesDict,
            self.dataSet.edgeTypeDecoderDict,
        )

    def getOptimizer(self, model: DecagonModel) -> DecagonOptimizer:
        optimizer = None
        with tf.name_scope('optimizer'):
            optimizer = DecagonOptimizer(
                model.embeddings,
                model.latent_inters,
                model.latent_varies,
                self.dataSet.degreesDict,
                self.dataSet.edgeTypeNumMatricesDict,
                self.dataSet.edgeTypeMatrixDimDict,
                self.dataSet.placeholdersDict,
                margin=self.dataSet.flags.max_margin,
                neg_sample_weights=self.dataSet.flags.neg_sample_size,
                batch_size=self.dataSet.flags.batch_size
            )

        return optimizer

    def getIterationResults(self) -> IterationResults:
        return IterationResults()

    def _recordTestEdges(self, dataSetIterator: EdgeMinibatchIterator) -> None:
        print('Starting recording test edges')
        f = open(self._getTestEdgeFilename(), 'w')
        writer = self._getWriter(f)

        for graphRelationType in dataSetIterator.graphAndRelationTypes:
            self._recordEdges(writer, graphRelationType, dataSetIterator)

        f.close()
        print('Ended recording test edges')

    def _getTestEdgeFilename(self) -> str:
        baseFname = self.config.getSetting('TestEdgeFilename')

        csvIdx = baseFname.find('.csv')
        if csvIdx != -1:
            baseFname = baseFname[:csvIdx]

        baseFname += '-%s' % datetime.datetime.now()
        baseFname = baseFname.replace(' ', '-')

        return baseFname + '.csv'

    def _recordEdges(self, dictWriter, graphRelationType, dataSetIterator) -> None:
        relTypeStr = ''
        if graphRelationType.graphType == DRUG_DRUG_GRAPH_TYPE:
            try:
                relTypeStr = SideEffectId.toDecagonFormat(
                    self.relIdxToRelId[graphRelationType.relationType]
                )
            except KeyError:
                # Will have a key error here for TPosed IDs. These test edges
                # are the same as those for which they are TPosed, so we just
                # write the non-transposed ones and don't process those transposed.
                return

        decoders = {
            0: ProteinId.toDecagonFormat,
            1: DrugId.toDecagonFormat,
        }

        nodeListDecoder = {
            0: self.nodeLists.proteinNodeList,
            1: self.nodeLists.drugNodeList,
        }

        def _getRecordDict(edge: Tuple[int, int], label: int):
            fromGraphType = graphRelationType.graphType[FROM_GRAPH_IDX]
            toGraphType = graphRelationType.graphType[TO_GRAPH_IDX]

            fromLen = 586 if fromGraphType == 1 else 17216
            toLen = 586 if fromGraphType == 1 else 17216

            if edge[FROM_NODE_IDX] >= fromLen or edge[TO_NODE_IDX] >= toLen:
                import pdb; pdb.set_trace()

            fromNodePre = nodeListDecoder[fromGraphType][edge[FROM_NODE_IDX]]
            toNodePre   = nodeListDecoder[toGraphType][edge[TO_NODE_IDX]]

            return {
                'FromNode': decoders[fromGraphType](fromNodePre),
                'ToNode': decoders[toGraphType](toNodePre),
                'RelationId': relTypeStr,
                'Label': label,
            }

        graphType    = graphRelationType.graphType
        relationType = graphRelationType.relationType

        posRecordDicts = map(
            lambda x: _getRecordDict(x, 1),
            dataSetIterator.val_edges[graphType][relationType]
        )

        negRecordDicts = map(
            lambda x: _getRecordDict(x, 0),
            dataSetIterator.val_edges_false[graphType][relationType]
        )

        # Write to writer
        list(map(lambda x: dictWriter.writerow(x), posRecordDicts))
        list(map(lambda x: dictWriter.writerow(x), negRecordDicts))

    def _getWriter(self, f) -> csv.DictWriter:
        fieldnames = [
            'FromNode',
            'ToNode',
            'RelationId',
            'Label'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        return writer


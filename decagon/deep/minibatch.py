from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp

from ..utility import preprocessing

np.random.seed(123)

EDGES_IDX = 0
DRUG_DRUG_GRAPH_TYPE = (1, 1)

class GraphRelationType:
    def __init__(self, graphType, relationType) -> None:
        self.graphType = graphType
        self.relationType = relationType

class EdgeMinibatchIterator(object):
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.
    assoc -- numpy array with target edges
    placeholders -- tensorflow placeholders object
    batch_size -- size of the minibatches
    """
    def __init__(self, adj_mats, feat, edge_types, drug_drug_test_edges, batch_size=100, val_test_size=0.01):
        self.adj_mats = adj_mats
        self.feat = feat
        self.edge_types = edge_types
        self.batch_size = batch_size
        self.val_test_size = val_test_size
        self.num_edge_types = sum(self.edge_types.values())
        self.drug_drug_test_edges = {
            i: test_edges
            for i, test_edges in enumerate(drug_drug_test_edges.values())
        }

        self.iter = 0
        self.freebatch_edge_types= list(range(self.num_edge_types))
        self.batch_num = [0]*self.num_edge_types
        self.current_edge_type_idx = 0
        self.edge_type2idx = {}
        self.idx2edge_type = {}
        self.adj_mtx_to_idx = {}
        r = 0
        for i, j in self.edge_types:
            for k in range(self.edge_types[i,j]):
                self.edge_type2idx[i, j, k] = r
                self.idx2edge_type[r] = i, j, k

                mtx = self.adj_mats[(i, j)][k]
                self.adj_mtx_to_idx[mtx.id] = ((i, j), k)

                r += 1

        self.train_edges = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        self.val_edges = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        self.test_edges = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        self.test_edges_false = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        self.val_edges_false = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}

        # Function to build test and val sets with val_test_size positive links
        self.adj_train = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        import pdb
        for i, j in self.edge_types:
            for k in range(self.edge_types[i,j]):
                print("Minibatch edge type:", "(%d, %d, %d)" % (i, j, k))
                self.mask_test_edges((i, j), k)

                print("Train edges=", "%04d" % len(self.train_edges[i,j][k]))
                print("Val edges=", "%04d" % len(self.val_edges[i,j][k]))
                print("Test edges=", "%04d" % len(self.test_edges[i,j][k]))

    @property
    def graphAndRelationTypes(self):
        for graphType, graphTypeMtxs in self.val_edges.items():
            for idx in range(len(graphTypeMtxs)):
                yield GraphRelationType(graphType, idx)

    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        if adj.shape[0] == adj.shape[1]:
            adj_ = adj + sp.eye(adj.shape[0])
            rowsum = np.array(adj_.sum(1))
            degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
            adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        else:
            rowsum = np.array(adj.sum(1))
            colsum = np.array(adj.sum(0))
            rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
            coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
            adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()
        return preprocessing.sparse_to_tuple(adj_normalized)

    def _ismember(self, a, b):
        a = np.array(a)
        b = np.array(b)
        rows_close = np.all(a - b == 0, axis=1)
        return np.any(rows_close)

    def _getIdxPairs(self, setShape):
        xx, yy = np.indices(setShape, dtype=np.dtype(np.int32))

        return np.dstack([xx, yy]).reshape(-1, 2)

    def _setDiffIndices(self, allIdxPairs, setPairsSubset):
        allPairsRows = allIdxPairs.view(
            [('', allIdxPairs.dtype)] * allIdxPairs.shape[1]
        )

        subsetPairsRows = setPairsSubset.view(
            [('', setPairsSubset.dtype)] * setPairsSubset.shape[1]
        )

        return np.setdiff1d(allPairsRows, subsetPairsRows).view(
            allIdxPairs.dtype
        ).reshape((-1, allPairsRows.shape[1]))


    def mask_test_edges(self, edge_type, type_idx):
        mtx = self.adj_mats[edge_type][type_idx]

        if mtx.isTranspose and self._test_edges_exist_tposed_mtx(mtx):
            return self._mask_test_edges_from_tpose(edge_type, type_idx, mtx)
        elif edge_type == DRUG_DRUG_GRAPH_TYPE and type_idx in self.drug_drug_test_edges:
            return self._mask_test_edges_drug_drug_precomputed(type_idx)
        else:
            return self._mask_test_edges_new(edge_type, type_idx)

    def _test_edges_exist_tposed_mtx(self, mtx):
        if not mtx.isTranspose:
            return False

        edge_type, type_idx = self.adj_mtx_to_idx[mtx.transposedMtxLink.id]
        return self.train_edges[edge_type][type_idx] is not None

    def _mask_test_edges_from_tpose(self, edge_type, type_idx, mtx):
        if not mtx.isTranspose:
            return False

        edge_type_tpose, type_idx_tpose = self.adj_mtx_to_idx[mtx.transposedMtxLink.id]

        old_train_infos = self.adj_train[edge_type_tpose][type_idx_tpose]
        new_edges = np.flip(old_train_infos[EDGES_IDX], axis=1)
        self.adj_train[edge_type][type_idx] = (
            new_edges,
            old_train_infos[1],
            (old_train_infos[2][1], old_train_infos[2][0])
        )

        self.train_edges[edge_type][type_idx] = np.flip(
            self.train_edges[edge_type_tpose][type_idx_tpose], axis=1
        )

        self.val_edges[edge_type][type_idx] = np.flip(
            self.val_edges[edge_type_tpose][type_idx_tpose], axis=1
        )

        self.val_edges_false[edge_type][type_idx] = np.flip(
            self.val_edges_false[edge_type_tpose][type_idx_tpose], axis=1
        )

        self.test_edges[edge_type][type_idx] = np.flip(
            self.test_edges[edge_type_tpose][type_idx_tpose], axis=1
        )

        try:
            self.test_edges_false[edge_type][type_idx] = np.flip(
                self.test_edges_false[edge_type_tpose][type_idx_tpose], axis=1
            )
        except ValueError:
            self.test_edges_false[edge_type][type_idx] = []

    def _mask_test_edges_new(self, edge_type, type_idx):
        edges_all, _, _ = preprocessing.sparse_to_tuple(self.adj_mats[edge_type][type_idx])
        num_test = max(50, int(np.floor(edges_all.shape[0] * 0)))#self.val_test_size)))
        num_val = max(50, int(np.floor(edges_all.shape[0] * self.val_test_size)))

        all_edge_idx = list(range(edges_all.shape[0]))
        np.random.shuffle(all_edge_idx)

        val_edge_idx = all_edge_idx[:num_val]
        val_edges = edges_all[val_edge_idx]

        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges_all[test_edge_idx]

        train_edges = np.delete(edges_all, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            if len(test_edges_false) % 1000 == 0:
                print("Constructing test edges=", "%04d/%04d" % (len(test_edges_false), len(test_edges)))
            idx_i = np.random.randint(0, self.adj_mats[edge_type][type_idx].shape[0])
            idx_j = np.random.randint(0, self.adj_mats[edge_type][type_idx].shape[1])
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if self._ismember([idx_i, idx_j], test_edges_false):
                    continue
            test_edges_false.append([idx_i, idx_j])
            break

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            if len(val_edges_false) % 1000 == 0:
                print("Constructing val edges=", "%04d/%04d" % (len(val_edges_false), len(val_edges)))
            idx_i = np.random.randint(0, self.adj_mats[edge_type][type_idx].shape[0])
            idx_j = np.random.randint(0, self.adj_mats[edge_type][type_idx].shape[1])
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if val_edges_false:
                if self._ismember([idx_i, idx_j], val_edges_false):
                    continue
            val_edges_false.append([idx_i, idx_j])
            break

        # Re-build adj matrices
        data = np.ones(train_edges.shape[0])
        adj_train = sp.csr_matrix(
            (data, (train_edges[:, 0], train_edges[:, 1])),
            shape=self.adj_mats[edge_type][type_idx].shape)
        self.adj_train[edge_type][type_idx] = self.preprocess_graph(adj_train)

        self.train_edges[edge_type][type_idx] = train_edges
        self.val_edges[edge_type][type_idx] = val_edges
        self.test_edges[edge_type][type_idx] = test_edges

        self.val_edges_false[edge_type][type_idx] = \
            np.array(val_edges_false).reshape((len(val_edges_false), 2))

        self.test_edges_false[edge_type][type_idx] = \
            np.array(test_edges_false).reshape((len(test_edges_false), 2))

    def _mask_test_edges_drug_drug_precomputed(self, type_idx):
        thisAdjMat = self.adj_mats[DRUG_DRUG_GRAPH_TYPE][type_idx]

        self.adj_train[DRUG_DRUG_GRAPH_TYPE][type_idx] = \
            self.preprocess_graph(thisAdjMat)

        self.train_edges[DRUG_DRUG_GRAPH_TYPE][type_idx] = np.dstack([
            thisAdjMat.nonzero()[0],
            thisAdjMat.nonzero()[1],
        ]).reshape(-1, 2)

        self.val_edges[DRUG_DRUG_GRAPH_TYPE][type_idx] = \
            self.drug_drug_test_edges[type_idx]['positive']

        self.val_edges_false[DRUG_DRUG_GRAPH_TYPE][type_idx] = \
            self.drug_drug_test_edges[type_idx]['negative']

        self.test_edges[DRUG_DRUG_GRAPH_TYPE][type_idx] = np.empty((0, 2))
        self.test_edges_false[DRUG_DRUG_GRAPH_TYPE][type_idx] = np.empty((0, 2))

    def end(self):
        finished = len(self.freebatch_edge_types) == 0
        return finished

    def update_feed_dict(self, feed_dict, dropout, placeholders):
        # construct feed dictionary
        feed_dict.update({
            placeholders['adj_mats_%d,%d,%d' % (i,j,k)]: self.adj_train[i,j][k]
            for i, j in self.edge_types for k in range(self.edge_types[i,j])})
        feed_dict.update({placeholders['feat_%d' % i]: self.feat[i] for i, _ in self.edge_types})
        feed_dict.update({placeholders['dropout']: dropout})

        return feed_dict

    def batch_feed_dict(self, batch_edges, batch_edge_type, placeholders):
        feed_dict = dict()
        feed_dict.update({placeholders['batch']: batch_edges})
        feed_dict.update({placeholders['batch_edge_type_idx']: batch_edge_type})
        feed_dict.update({placeholders['batch_row_edge_type']: self.idx2edge_type[batch_edge_type][0]})
        feed_dict.update({placeholders['batch_col_edge_type']: self.idx2edge_type[batch_edge_type][1]})

        return feed_dict

    def next_minibatch_feed_dict(self, placeholders):
        """Select a random edge type and a batch of edges of the same type"""
        modVal = 4 if (1, 0, 0) in self.edge_type2idx else 3
        while True:
            if self.iter % modVal == 0:
                # gene-gene relation
                self.current_edge_type_idx = self.edge_type2idx[0, 0, 0]
            elif self.iter % modVal == 1:
                # gene-drug relation
                self.current_edge_type_idx = self.edge_type2idx[0, 1, 0]
            elif self.iter % modVal == 2 and modVal == 4:
                # drug-gene relation
                self.current_edge_type_idx = self.edge_type2idx[1, 0, 0]
            else:
                # random side effect relation
                if len(self.freebatch_edge_types) > 0:
                    self.current_edge_type_idx = np.random.choice(self.freebatch_edge_types)
                else:
                    self.current_edge_type_idx = self.edge_type2idx[0, 0, 0]
                    self.iter = 0

            i, j, k = self.idx2edge_type[self.current_edge_type_idx]
            if self.batch_num[self.current_edge_type_idx] * self.batch_size \
                   <= len(self.train_edges[i,j][k]) - self.batch_size:
                break
            else:
                if self.iter % 4 in [0, 1, 2]:
                    self.batch_num[self.current_edge_type_idx] = 0
                else:
                    self.freebatch_edge_types.remove(self.current_edge_type_idx)

        self.iter += 1
        start = self.batch_num[self.current_edge_type_idx] * self.batch_size
        self.batch_num[self.current_edge_type_idx] += 1
        batch_edges = self.train_edges[i,j][k][start: start + self.batch_size]
        return self.batch_feed_dict(batch_edges, self.current_edge_type_idx, placeholders)

    def num_training_batches(self, edge_type, type_idx):
        return len(self.train_edges[edge_type][type_idx]) // self.batch_size + 1

    def val_feed_dict(self, edge_type, type_idx, placeholders, size=None):
        edge_list = self.val_edges[edge_type][type_idx]
        if size is None:
            return self.batch_feed_dict(edge_list, edge_type, placeholders)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges, edge_type, placeholders)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        for edge_type in self.edge_types:
            for k in range(self.edge_types[edge_type]):
                self.train_edges[edge_type][k] = np.random.permutation(self.train_edges[edge_type][k])
                self.batch_num[self.edge_type2idx[edge_type[0], edge_type[1], k]] = 0
        self.current_edge_type_idx = 0
        self.freebatch_edge_types = list(range(self.num_edge_types))

        edgeTypesToRemove = [(0, 0, 0), (0, 1, 0), (1, 0, 0)]
        for edgeTypeToRemove in edgeTypesToRemove:
            if edgeTypeToRemove in self.edge_type2idx:
                self.freebatch_edge_types.remove(
                    self.edge_type2idx[edgeTypeToRemove]
                )

        self.iter = 0

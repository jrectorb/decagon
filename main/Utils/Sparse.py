from typing import Type
import scipy.sparse as sp

class RelationCooMatrix(sp.coo_matrix):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.isTranspose: bool = False

        # A reference to the transposed matrix
        self.transposedMtxLink: RelationCooMatrix = None

    def transpose(self, axes=None, copy=False) -> Type['RelationCooMatrix']:
        newMtx = RelationCooMatrix(super().transpose(axes, copy))

        self.isTranspose = True
        self.tranposedMtxLink = newMtx

        newMtx.isTranspose = True
        newMtx.transposedMtxLink = self

        return newMtx

    def tocsr(self, copy=False) -> Type['RelationCsrMatrix']:
        newMtx = RelationCsrMatrix(super().tocsr(copy))

        newMtx.isTranspose = self.isTranspose
        newMtx.transposedMtxLink = self.transposedMtxLink

        return newMtx

class RelationCsrMatrix(sp.csr_matrix):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.isTranspose: bool = False

        # A reference to the transposed matrix
        self.transposedMtxLink: RelationCsrMatrix = None

    def transpose(self, axes=None, copy=False) -> Type['RelationCsrMatrix']:
        newMtx = RelationCsrMatrix(super().transpose(axes, copy))

        self.isTranspose = True
        self.transposedMtxLink = newMtx

        newMtx.isTranspose = True
        newMtx.transposedMtxLink = self

        return newMtx

    def tocoo(self, copy=False) -> Type['RelationCooMatrix']:
        newMtx = RelationCooMatrix(super().tocoo(copy))

        newMtx.isTranspose = self.isTranspose
        newMtx.transposedMtxLink = self.transposedMtxLink

        return newMtx


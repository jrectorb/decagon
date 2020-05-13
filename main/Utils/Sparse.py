import scipy.sparse as sp

class RelationCooMatrix(sp.coo_matrix):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.isTranspose = False

        # A reference to the transposed matrix
        self.transposedMtxLink = None

    def transpose(self, axes=None, copy=False) -> RelationCooMatrix:
        newMtx = RelationCooMatrix(super().transpose(axes, copy))

        self.isTranspose = True
        self.tranposedMtxLink = newMtx

        newMtx.isTranspose = True
        newMtx.transposedMtxLink = self

        return newMtx

class RelationCsrMatrix(sp.csr_matrix):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.isTranspose = False

        # A reference to the transposed matrix
        self.transposedMtxLink = None

    def transpose(self, axes=None, copy=False) -> RelationCsrMatrix:
        newMtx = RelationCsrMatrix(super().transpose(axes, copy))

        self.isTranspose = True
        self.tranposedMtxLink = newMtx

        newMtx.isTranspose = True
        newMtx.transposedMtxLink = self

        return newMtx


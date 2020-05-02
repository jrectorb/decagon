import scipy.sparse as sp

class DecagonCsrMatrix(sp.csr.csr_matrix):
    def __init__(self, matrixType, *args, **kwargs):
        self.matrixType = matrixType
        super().__init__(*args, **kwargs)

from .main.Predictor.NpPredictor import NpPredictor, TrainingEdgeIterator
from sklearn import metrics
import sys

# Run me as python -m NpPredictorExample.ExampleRunner --config NpPredictorExample/configuration.json

def _get_importance_matrix(train_edges: np.ndarray) -> np.ndarrray:
    # Fill this part in
    pass

def _get_auc(predictions: np.ndarray) -> float:
    LABEL_COL_IDX     = 3
    PREDICTION_COL_IDX = 4

    return metrics.roc_auc_score(
        y_true=predictions[:, LABEL_COL_IDX],
        y_score=predictions[:, PREDICTION_COL_IDX]
    )

def main() -> int:
    # Let's say we want to test on Mumps, which has a STITCH Id of C0026780
    side_effect_id = 'C0026780'

    # First, get train edges specific to this side effect
    train_edge_iterator = TrainingEdgeIterator(side_effect_id)

    # train_edges is a 3-dim ndarray where the first column is the
    # from drug index, the second column is the to drug index,
    # and the third column is the label of the edge between them
    train_edges = train_edge_iterator.get_train_edges()

    # Now, do the model training and determine some square feature importance
    # matrix wherein the feature importance matrix is of size d x d.  Here, d
    # is the size of the node embeddings
    importance_mtx = _get_importance_matrix(train_edges)

    # Now, define the predictor for the same side effect Id
    predictor = NpPredictor(side_effect_id)

    # predictions is a 4-dim ndarray where the first column is the from
    # drug index, the second column is the to drug index, the third column
    # is the true label, and the fourth column is the predicted probability
    predictions = predictor.predict(importance_mtx)

    # Now, for example, print the AUC
    print('AUC: %f' % _get_auc(predictions))

    return 0

if __name__ == '__main__':
    sys.exit(main())


from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib as mpl
from lib.analyze.prediction import Prediction


def confusion_matrix_single(fold_prediction: Prediction):
    with mpl.rc_context({'figure.figsize': [5, 5]}):
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            fold_prediction.ground_truth,
            fold_prediction.thresholded,
            labels=(1.0, 0.0),
            display_labels=('Active', 'Inactive'),
            cmap=plt.cm.Blues,
            ax=ax
        )
        ax.set(
            title=f'Confusion matrix of {fold_prediction.name}'
        )

        return fig

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.metrics import confusion_matrix


# TODO reverse dependency between manager_util and predict

@dataclass
class Prediction:
    experiment_name: str
    fold_num: int

    drug_split_name: str
    target_split_name: str
    mode: str

    predictions: ndarray
    ground_truth: ndarray

    @property
    def thresholded(self) -> ndarray:
        return np.where(self.predictions > 0.5, 1.0, 0.0)

    @property
    def name(self):
        return f'{self.experiment_name}({self.fold_num})'

    def get_scores(self) -> pd.DataFrame:
        TN, FP, FN, TP = confusion_matrix(self.thresholded, self.ground_truth).ravel()
        TN, FP, FN, TP = int(TN), int(FP), int(FN), int(TP)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        return pd.DataFrame(
            {
                'True negatives': [TN],
                'False positives': [FP],
                'False negatives': [FN],
                'True positives': [TP],
                'Precision': [TP / (TP + FN)],
                'Recall': [TP / (TP + FN)],
                'Accuracy': [(TP + TN) / (TP + TN + FP + FN)],
                'Sensitivity': [sensitivity],
                'Specificity': [specificity],
                'Balanced accuracy': [(sensitivity + specificity) / 2],
                'Fall out': [FP / (FP + TN)],
                'Miss rate': [FN / (FN + TN)],
                'Matthews correlation coefficient': [
                    (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                ],
            },
            index=[self.name]
        )

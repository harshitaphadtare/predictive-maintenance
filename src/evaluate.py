from sklearn.metrics import mean_absolute_error, mean_squared_error,\
    r2_score,precision_score, recall_score, f1_score
import numpy as np


def regression_metrics(Y_test,Y_pred):
    return{
        "MAE:": mean_absolute_error(Y_test,Y_pred),
        "RMSE:": np.sqrt(mean_squared_error(Y_test, Y_pred)),
        "R2:": r2_score(Y_test,Y_pred)
    }

def classification_metrics(Y_test,Y_pred):
    return{
        "Precision:":precision_score(Y_test,Y_pred),
        "Recall:":recall_score(Y_test,Y_pred),
        "F1:":f1_score(Y_test,Y_pred)
    }


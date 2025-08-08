#regression models
from sklearn.linear_model import Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

#classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import is_classifier, is_regressor
from sklearn.feature_selection import SelectKBest
import numpy as np
import pandas as pd
import joblib

model_configs = {
    "ridge": {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(max_iter=5000))
        ]),
        "params": {
            "ridge__alpha": [0.1, 1.0, 10.0]
        }
    },

    "lasso": {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('lasso', Lasso(max_iter=5000))
        ]),
        "params": {
            "lasso__alpha": [0.1, 1.0, 10.0]
        }
    },

    "rf_reg": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10]
        }
    },

    "rf_clf": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10]
        }
    },

    "logistic_reg": {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('log_reg', LogisticRegression(max_iter=5000, solver='liblinear'))
        ]),
        "params": {
            "log_reg__C": [0.1, 1.0, 10.0]
        }
    },

    "svm": {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC())
        ]),
        "params": {
            "svc__C": [0.1, 1.0, 10.0],
            "svc__kernel": ["linear", "sigmoid", "rbf"]
        }
    },

    "polynomial": {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('ridge', Ridge(max_iter=5000))
        ]),
        "params": {
            "poly__degree": [2, 3],
            "ridge__alpha": [0.1, 1.0, 10.0]
        }
    }
}

#selection of top features
def select_top_features(X, y, model, k_range, score_func, scoring='neg_root_mean_squared_error', cv=5):
    best_score = -np.inf
    best_k = None
    best_features = None

    for k in k_range:
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        scores = cross_val_score(model, X_selected, y, scoring=scoring, cv=cv)
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_k = k
            if isinstance(X, pd.DataFrame):
                best_features = X.columns[selector.get_support()]
            else:
                best_features = selector.get_support(indices=True)
    
    return best_k, best_features


#training model
def train_model(X_train,Y_train,model,param_grid,task='regression'):
    
    if is_classifier(model):
        scoring = 'accuracy' 
    elif is_regressor(model):
        scoring = 'neg_mean_squared_error'
    else:
        raise ValueError("Unknown model type")

    grid = GridSearchCV(model,param_grid,cv=5,scoring=scoring,error_score='raise')
    grid.fit(X_train,Y_train)
    
    return grid.best_estimator_, grid.best_params_, grid.best_score_

#saving model
def save_model(model,name):
    joblib.dump(model,f"models/{name}.joblib")



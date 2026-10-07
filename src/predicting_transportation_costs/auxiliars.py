import pandas as pd
import sklearn as skl
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.cluster import KMeans
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from predicting_transportation_costs.pipeline_steeps import feature_engineering
from predicting_transportation_costs.pipeline_steeps import cleaning
from predicting_transportation_costs import config as cfg
import yaml
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import  make_scorer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def get_pipeline_target_log_transform(pipeline : Pipeline) -> Pipeline:
    pipeline_with_transform = TransformedTargetRegressor(
    regressor=pipeline,
    func=np.log1p,     
    inverse_func=np.expm1
    )
    return pipeline_with_transform

def get_pipeline(estimator : BaseEstimator, features_dict : dict, scale_features : bool=False, n_clusters : int = 5) -> Pipeline :

    categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    numerical_steps= [("imputer", SimpleImputer(strategy="median"))]

    if(scale_features) : 
        numerical_steps.append(("scaler", StandardScaler()))
        
    numerical_transformer = Pipeline(steps=numerical_steps)

    NUMERIC_FEATURES= features_dict["features"]["numeric"] 
    CATEGORICAL_FEATURES= features_dict["features"]["categorical"] 


    preprocessor = ColumnTransformer(transformers= [
            ("num", numerical_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ],
            remainder="passthrough")
  

    pipeline = Pipeline(steps=[("cleaning",cleaning()),("feature_engineering", feature_engineering(n_clusters)),("preprocessor",preprocessor), ("model",estimator)])
    return pipeline


def get_features_dict() -> dict :
    with open(cfg.CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def prepare_data(df : pd.DataFrame) -> tuple[pd.DataFrame, pd.Series] :
    df_aux = df.drop(columns=["load_id"])
    Y = df_aux["posted_rate"]
    X = df_aux.drop(columns=["posted_rate"])
    return X,Y
        
def get_linear_regresion_baseline() -> Ridge:
    baseline_model = Ridge(
    alpha=1.0,
    solver="auto",
    fit_intercept=True,
    random_state=42
    )
    return baseline_model

def get_xgb_baseline() -> XGBRegressor:
    xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    objective='reg:absoluteerror', 
    eval_metric='mae',
    min_child_weight=30,       
    subsample=0.8,            
    colsample_bytree=0.8,     
    gamma=0.1,                
    random_state=42,
    n_jobs=-1
    )
    return xgb_model

def get_lgbm_baseline() -> LGBMRegressor:
    lgbm_model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    objective='regression_l1',
    max_depth=6,
    num_leaves=31,            
    min_child_samples=20,     
    subsample=0.8,           
    colsample_bytree=0.8,     
    random_state=42,
    n_jobs=-1
    )
    return lgbm_model

def get_random_forest_baseline() -> RandomForestRegressor:
    random_forest_model = RandomForestRegressor(n_estimators=100,     
    max_depth=12,           
    min_samples_split=10,   
    min_samples_leaf=5,     
    max_features="sqrt",    
    random_state=42,        
    n_jobs=-1
    )
    return random_forest_model

def get_final_pipeline() -> Pipeline:
   model= Ridge(alpha=17.269978740046007, random_state=42)
   pipeline= get_pipeline(model,get_features_dict(),True,n_clusters=5)
   return pipeline



def run_temporal_cv(pipeline : Pipeline, X_train : pd.DataFrame, y_train : pd.DataFrame, show_logs : bool =True, eval_metric : str = "RMSE") -> np.ndarray :

    scoring_map = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
    }

    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=tscv,
    scoring= scoring_map[eval_metric],
    n_jobs=-1
    )

    nmae_scores = -cv_scores

    if(show_logs) :
        print("results per fold:")
        for i, score in enumerate(nmae_scores, 1):
            print(f"  Fold {i}: {score:.4f}")

        print("------")
        print(f"  {eval_metric} Mean: {nmae_scores.mean():.4f}")
        print(f"  Std: {nmae_scores.std():.4f}")
        print(f"  Min / Max: [{nmae_scores.min():.4f}, {nmae_scores.max():.4f}]")
    return -cv_scores

def show_results(y_test : np.ndarray, y_predicted : np.ndarray) -> None :
    test_mae= mean_absolute_error(y_true=y_test,y_pred=y_predicted)
    test_rsme= root_mean_squared_error(y_true=y_test,y_pred=y_predicted)
    print("----------------")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test RMSE: {test_rsme:.4f}")
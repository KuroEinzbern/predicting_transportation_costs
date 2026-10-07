import pandas as pd
import numpy as np
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
from lightgbm import LGBMClassifier

from xgboost import XGBRegressor



class feature_engineering(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = None

    
    def fit(self, X, y=None):



        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=42, 
            n_init=10
        )

        coords = X[["pickup_lat", "pickup_lon"]].values
        self.kmeans.fit(coords)

        return self

    def transform(self, X):
        df = X.copy()

        #missing flag
        df["weight_is_missing"] = df["weight"].isna().astype(int)

       

        #date feature engineering
        df["day_of_week"] = df["date"].dt.day_name().astype("category")
        df["is_weekend"] = ((df["day_of_week"] == "Saturday") | (df["day_of_week"] == "Sunday")).astype(int)
        df["month"] = (df["date"].dt.month).astype("category")
        df["n_day"] = df["date"].dt.day

        #cloustering zones and zone feature engineering

        df["zone_clouster_pickup"] = np.nan
        df["zone_clouster_delivery"] = np.nan
        df["route"]  = np.nan

        if  (df["pickup_lat"].isna().sum() == 0) & (df["pickup_lon"].isna().sum() == 0) :
            pickup_coords = df[["pickup_lat", "pickup_lon"]].values
            df["zone_clouster_pickup"] = self.kmeans.predict(pickup_coords).astype(str)
            pickup_coords = df[["delivery_lat", "delivery_lon"]].values
            df["zone_clouster_delivery"] = self.kmeans.predict(pickup_coords).astype(str)
            df["route"]  = df["zone_clouster_pickup"] + "_" + df["zone_clouster_delivery"]


        #interactions
        df["index_x_distance"] = df["market_index"] * df["distance"]
        df["distance_x_weight"] = df["distance"] * df["weight"]
   #     df["index_x_signal"] = df["market_index"] * df["quote_signal"]

        #threshold = 1600
        #df["distance_long_haul"] = np.maximum(0, df["distance"] - threshold)

        #drops
        df = df.drop(columns=["date"])    #"pickup_lat","pickup_lon","delivery_lat","delivery_lon"

        
        return df


class cleaning(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
     return self

    def transform(self, X): 
        #casting types
        X["date"] = pd.to_datetime(X["date"], errors="coerce")
        X["pickup"] = X["pickup"].astype("category")
        X["delivery"] = X["delivery"].astype("category")
        X["equipment"] = X["equipment"].astype("category")
        X["delivery_lat"] = pd.to_numeric(X["delivery_lat"], errors="coerce").astype("float64")
      
        #casting into NAN the invalid values in weight
        X["weight"]= X["weight"].mask(X["weight"]<= 0,np.nan)
        return X

    

    
        
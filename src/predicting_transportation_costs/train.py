import pandas as pd
from predicting_transportation_costs import config as cfg
from predicting_transportation_costs.auxiliars import prepare_data,get_final_pipeline
import joblib
import os
from dotenv import load_dotenv

def main():

    load_dotenv()
    model_version = os.getenv("model_version", 1.0)

    train_df= pd.read_parquet(cfg.DATA_DIR / "train.parquet")
    test_df= pd.read_parquet(cfg.DATA_DIR / "test.parquet")

    #to use all the data avaible for the final model
    full_train_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    X_train, Y_train = prepare_data(full_train_df)
    pipeline = get_final_pipeline()
    pipeline.fit(X_train,Y_train)
    joblib.dump(pipeline,filename=cfg.MODEL_DIR / f"model_{model_version}")
    
if __name__ == "__main__":
    main()


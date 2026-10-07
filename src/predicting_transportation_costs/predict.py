import pandas as pd
import numpy as np
from predicting_transportation_costs import config as cfg
from predicting_transportation_costs.auxiliars import prepare_data,get_final_pipeline
import joblib


def main():
    validation_df = pd.read_csv(cfg.DATA_DIR / "validation.csv")
    train_df = pd.read_csv(cfg.DATA_DIR / "train.csv")
    X_train , _ =prepare_data(train_df)

    pipeline= joblib.load(cfg.MODEL_DIR / "model_1.0")
    ids= validation_df["load_id"]
    X_val = validation_df.drop(columns= ["load_id"])
    predictions= pipeline.predict(X=X_val)
    pred_to_df= pd.Series(predictions)
    validation_predictions= pd.DataFrame({"load_id":ids,"predicted_rate": pred_to_df})
    validation_predictions.to_csv(cfg.PROJECT_ROOT / "validation_predictions.csv",index=False)

    december_df = pd.read_csv(cfg.DATA_DIR / "december-chart-inputs.csv")
    december_df_to_train= december_df.copy()
    expected_columns = X_train.columns

    for col in expected_columns:
        if col not in december_df_to_train.columns:
            december_df_to_train[col] = (np.nan)

    x_december= december_df_to_train.drop(columns=["predicted_rate"])
    december_predicted_rate= pipeline.predict(x_december)
    december_df["predicted_rate"] = december_predicted_rate
    december_df.to_csv(cfg.DATA_DIR / "december_chart_inputs.csv",index=False)

if __name__ == "__main__":
    main()
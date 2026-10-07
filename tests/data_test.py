import pandas as pd
import predicting_transportation_costs.config as cfg


def test_training_data_contains_expected_columns():
    df = pd.read_csv(cfg.DATA_DIR / "train-test.csv")

    expected_columns = {
        "load_id",
        "pickup",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "equipment",
        "weight",
        "date",
        "market_index",
        "quote_signal",
        "posted_rate"
    }
    assert expected_columns.issubset(df.columns)


def test_temporal_split():

    df_train = pd.read_parquet(cfg.DATA_DIR / "train.parquet")
    df_test = pd.read_parquet(cfg.DATA_DIR / "test.parquet")

    #no overlaping
    train_ids= df_train["load_id"]
    test_ids= df_test["load_id"]
    assert (train_ids.isin(test_ids)).sum() == 0

    #proportion 
    assert len(df_train) > (len(df_test) * 3) 

    #correct_temporal_separation
    assert df_train["date"].max()  < df_test["date"].min()

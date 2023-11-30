import pandas as pd


def create_best_model_df(data):
    best_model_df_train = pd.DataFrame(
        {
            "dates_train": data["dates_train"],
            "y_train": data["y_train"],
            "y_pred_train": data["y_pred_train"],
        }
    )
    best_model_df_test = pd.DataFrame(
        {
            "dates_test": data["dates_test"],
            "y_test": data["y_test"],
            "y_pred_test": data["y_pred_test"],
        }
    )
    best_model_df_train.to_pickle(data["output_path_train"])
    best_model_df_test.to_pickle(data["output_path_test"])

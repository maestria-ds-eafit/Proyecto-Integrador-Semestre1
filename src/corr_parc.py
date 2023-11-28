import pingouin as pg
import pandas as pd


def partial_correlation(df, x, y, others, method="pearson"):
    return pg.partial_corr(data=df, x=x, y=y, covar=others, method=method)


def calculate_partial_correlation_df(df):
    partial_correlations = []
    variables = []
    exogen_data = df.drop(columns=["energy_price"])

    for x in list(exogen_data.columns):
        temp_r = partial_correlation(
            df,
            x,
            "energy_price",
            list(df.drop(columns=["energy_price", x])),
        )["r"].iloc[0]
        partial_correlations.append(temp_r)
        variables.append(x)

    df_corr_parc = pd.DataFrame(
        {"Variable": variables, "partial_correlation": partial_correlations}
    )
    df_corr_parc["abs_partial_correlation"] = df_corr_parc["partial_correlation"].abs()
    df_corr_parc.sort_values(by="abs_partial_correlation", ascending=False)[
        ["Variable", "abs_partial_correlation"]
    ].style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1)
    return df_corr_parc


def apply_partial_correlation_criteria(data):
    correlation_matrix = data["correlation_matrix"]
    partial_correlation_df = data["partial_correlation_df"]
    excluded = set(data.get("excluded", []))
    threshold = data.get("threshold", 0.5)

    columns_to_drop = set()
    columns_to_keep = set(correlation_matrix.columns)
    excluded_columns = set()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            col_i = correlation_matrix.columns[i]
            col_j = correlation_matrix.columns[j]

            # Check if either column is in the excluded list
            if col_i in excluded or col_j in excluded:
                continue

            # Check correlation and decide which column to keep
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                # Look up partial correlations for both variables
                partial_corr_i = partial_correlation_df.loc[
                    partial_correlation_df["Variable"] == col_i,
                    "abs_partial_correlation",
                ].iloc[0]
                partial_corr_j = partial_correlation_df.loc[
                    partial_correlation_df["Variable"] == col_j,
                    "abs_partial_correlation",
                ].iloc[0]

                # Decide which to drop based on partial correlation
                if partial_corr_i > partial_corr_j:
                    columns_to_drop.add(col_j)
                else:
                    columns_to_drop.add(col_i)
            else:
                columns_to_keep.update([col_i])

    # Update the columns to keep by removing the ones to drop
    columns_to_keep -= columns_to_drop

    return list(columns_to_keep), list(excluded_columns)

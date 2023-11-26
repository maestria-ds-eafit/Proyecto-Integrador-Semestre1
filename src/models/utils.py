import pandas as pd


def get_significant_variables(result_summary):
    result_summary_df = result_summary.tables[1]
    significant_variables = []
    significance_level = 0.05
    for i in range(1, len(result_summary_df)):
        p_value = float(result_summary_df[i][4].data)
        if p_value < significance_level:
            significant_variables.append(result_summary_df[i][0].data)
    #print(pd.DataFrame({"Variables significativas": significant_variables}))
    return significant_variables
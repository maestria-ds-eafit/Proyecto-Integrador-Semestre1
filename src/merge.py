import pandas as pd


def generate_merged_file():
    df_brent = pd.read_csv("./processed_tables/brent.csv")
    df_precios = pd.read_csv("./processed_tables/precios.csv")
    df_precipitacion = pd.read_csv("./processed_tables/precipitacion_represas.csv")
    df_temperatura = pd.read_csv("./processed_tables/temperatura_represas.csv")
    df_trm = pd.read_csv("./processed_tables/TRM.csv")
    df_brent = df_brent.rename(columns={"Fecha": "Date"})
    df_precipitacion = df_precipitacion.rename(columns={"date": "Date"})
    df_temperatura = df_temperatura.rename(columns={"date": "Date"})
    df_trm = df_trm.rename(columns={"vigenciadesde": "Date"})

    # Perform left joins
    df_merged = pd.merge(df_precios, df_brent, on="Date", how="left")
    df_merged = pd.merge(df_merged, df_precipitacion, on="Date", how="left")
    df_merged = pd.merge(df_merged, df_temperatura, on="Date", how="left")
    df_merged = pd.merge(df_merged, df_trm, on="Date", how="left")

    return df_merged


def export_merged_file():
    df_merged = generate_merged_file()
    df_merged.to_csv("./processed_tables/merged.csv", index=False)

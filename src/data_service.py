from azure.data.tables import TableClient
from azure.storage.blob import BlobServiceClient
import pandas as pd
import io


# Crear conexión de una tabla específica dentro del servicio de Azure Table Storage
def set_table_service(connection_string, table):
    """Crear servicio de conexión a Azure Table Storage"""
    return TableClient.from_connection_string(
        conn_str=connection_string, table_name=table
    )


# Obtener datos de Table Storage
def get_data_from_table_storage_table(table_service, filter_query):
    """Recuperar datos de Table Storage"""
    for record in table_service.query_entities(filter_query):
        yield record


# Crear DataFrame con los datos de la tabla consultada
def get_dataframe_from_table_storage_table(table_service, filter_query):
    """Crear un DataFrame con la data del Table Storage"""
    df = pd.DataFrame(get_data_from_table_storage_table(table_service, filter_query))
    df.drop(columns=["PartitionKey", "RowKey"], inplace=True)
    return df


def obtain_content_of_blob(connection_string, container_name, file_name):
    """
    Esta función crea una conexión a un archivo específico almacenado en blob Storage y retorna su contenido
    """
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(file_name)
    blob_stream = blob_client.download_blob().readall()
    blob_file = io.BytesIO(blob_stream)
    return blob_file

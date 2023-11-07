CREATE EXTERNAL TABLE TRM (
    [valor] float,
    [vigenciadesde] date,
    [vigenciahasta] date
)   WITH (
        LOCATION = '/*.csv',
        DATA_SOURCE = TRMFiles,
        FILE_FORMAT = CSVFormat
    );

SELECT
    TOP 100 *
FROM
    OPENROWSET(
        BULK '/*.csv',
        DATA_SOURCE = 'TRMFiles',
        FORMAT = 'CSV',
        PARSER_VERSION = '2.0',
        HEADER_ROW = TRUE
    ) with (
        valor float 1,
        vigenciadesde date 2,
        vigenciahasta date 3
    ) AS rows
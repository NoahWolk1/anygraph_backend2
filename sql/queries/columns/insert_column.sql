INSERT INTO "column" (email, name, datatype, example_value, dataset_url)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (name, dataset_url)
DO UPDATE SET
    datatype = EXCLUDED.datatype,
    example_value = EXCLUDED.example_value
RETURNING email, name, datatype, example_value, dataset_url;

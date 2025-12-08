SELECT email, name, datatype, example_value, dataset_url
FROM "column"
WHERE dataset_url = %s
ORDER BY name;

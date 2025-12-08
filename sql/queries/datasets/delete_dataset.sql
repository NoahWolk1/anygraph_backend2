DELETE FROM datasets
WHERE dataset_url = %s
RETURNING dataset_url;

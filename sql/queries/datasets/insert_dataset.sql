INSERT INTO datasets (dataset_url, name, file_type, chat_session_id, uploaded_at)
VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
ON CONFLICT (dataset_url)
DO UPDATE SET
    name = EXCLUDED.name,
    file_type = EXCLUDED.file_type
RETURNING dataset_url, name, file_type, uploaded_at, chat_session_id;

SELECT dataset_url, name, file_type, uploaded_at, chat_session_id
FROM datasets
WHERE dataset_url = %s;

SELECT dataset_url, name, file_type, uploaded_at, chat_session_id
FROM datasets
WHERE chat_session_id = %s
ORDER BY uploaded_at DESC;

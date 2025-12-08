SELECT COUNT(*) as total_datasets
FROM datasets d
JOIN chat_sessions cs ON d.chat_session_id = cs.chat_session_id
WHERE cs.email = %s;

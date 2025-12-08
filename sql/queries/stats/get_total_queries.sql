SELECT COUNT(*) as total_queries
FROM messages m
JOIN chat_sessions cs ON m.chat_session_id = cs.chat_session_id
WHERE cs.email = %s AND m.sender = 'user';

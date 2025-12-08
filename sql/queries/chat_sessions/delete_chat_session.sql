DELETE FROM chat_sessions
WHERE chat_session_id = %s
RETURNING chat_session_id;

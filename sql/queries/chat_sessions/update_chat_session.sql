UPDATE chat_sessions
SET chat_session_title = %s
WHERE chat_session_id = %s
RETURNING chat_session_id, email, chat_session_title, created_at;

SELECT chat_session_id, email, chat_session_title, created_at
FROM chat_sessions
WHERE chat_session_id = %s;

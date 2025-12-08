INSERT INTO chat_sessions (email, chat_session_title, created_at)
VALUES (%s, %s, CURRENT_TIMESTAMP)
RETURNING chat_session_id, email, chat_session_title, created_at;

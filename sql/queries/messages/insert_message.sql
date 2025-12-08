INSERT INTO messages (sender, message_txt, chat_session_id, generated_code, created_at)
VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
RETURNING message_id, sender, message_txt, created_at, chat_session_id, generated_code;

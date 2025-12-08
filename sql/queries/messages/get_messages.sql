SELECT message_id, sender, message_txt, created_at, chat_session_id, generated_code
FROM messages
WHERE chat_session_id = %s
ORDER BY created_at ASC;

SELECT COUNT(*) as sessions_this_month
FROM chat_sessions
WHERE email = %s AND created_at >= %s;

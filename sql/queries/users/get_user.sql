SELECT email, user_id, full_name, created_at, last_log_in
FROM "user"
WHERE email = %s;

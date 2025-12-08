UPDATE "user"
SET
    full_name = COALESCE(%s, full_name),
    last_log_in = COALESCE(%s, last_log_in)
WHERE email = %s
RETURNING email, user_id, full_name, created_at, last_log_in;

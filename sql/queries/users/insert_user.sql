INSERT INTO "user" (email, full_name, created_at, last_log_in)
VALUES (%s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (email)
DO UPDATE SET
    last_log_in = CURRENT_TIMESTAMP,
    full_name = COALESCE(EXCLUDED.full_name, "user".full_name)
RETURNING email, user_id, full_name, created_at, last_log_in;

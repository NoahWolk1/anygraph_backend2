CREATE TABLE IF NOT EXISTS "user" (
    email VARCHAR(255) PRIMARY KEY,
    user_id UUID DEFAULT gen_random_uuid() NOT NULL UNIQUE,
    full_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_log_in TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_user_id ON "user"(user_id);

CREATE TABLE IF NOT EXISTS chat_sessions (
    chat_session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL,
    chat_session_title VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (email) REFERENCES "user"(email) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_email ON chat_sessions(email);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created_at ON chat_sessions(created_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sender VARCHAR(50) NOT NULL CHECK (sender IN ('user', 'assistant', 'system')),
    message_txt TEXT NOT NULL,
    generated_code TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chat_session_id UUID NOT NULL,
    FOREIGN KEY (chat_session_id) REFERENCES chat_sessions(chat_session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_session_id ON messages(chat_session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_url VARCHAR(1000) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL CHECK (file_type IN ('csv', 'excel', 'xlsx', 'xls')),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chat_session_id UUID NOT NULL,
    FOREIGN KEY (chat_session_id) REFERENCES chat_sessions(chat_session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_datasets_chat_session_id ON datasets(chat_session_id);

CREATE TABLE IF NOT EXISTS "column" (
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    datatype VARCHAR(100) NOT NULL,
    example_value TEXT,
    dataset_url VARCHAR(1000) NOT NULL,
    PRIMARY KEY (name, dataset_url),
    FOREIGN KEY (email) REFERENCES "user"(email) ON DELETE CASCADE,
    FOREIGN KEY (dataset_url) REFERENCES datasets(dataset_url) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_column_dataset_url ON "column"(dataset_url);
CREATE INDEX IF NOT EXISTS idx_column_email ON "column"(email);

CREATE TABLE IF NOT EXISTS share_urls (
    shared_url VARCHAR(500) PRIMARY KEY,
    chat_session_id UUID NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chat_session_id) REFERENCES chat_sessions(chat_session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_share_urls_chat_session_id ON share_urls(chat_session_id);

COMMENT ON TABLE "user" IS 'Stores user authentication and profile information';
COMMENT ON TABLE chat_sessions IS 'Chat sessions where users interact with the AI for data analysis';
COMMENT ON TABLE messages IS 'Individual messages within chat sessions';
COMMENT ON TABLE datasets IS 'Uploaded datasets for analysis';
COMMENT ON TABLE "column" IS 'Column metadata extracted from datasets including data types';
COMMENT ON TABLE share_urls IS 'Shareable links for chat sessions';

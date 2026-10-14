-- DeepGuard Multimodal Detection System
-- SQLite Database Schema Definition
-- Active Production DB: backend/instance/deepguard.db

-- 1. Users Table: Account credentials, profiles & authentication
CREATE TABLE IF NOT EXISTS users (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(120) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. Predictions Table: Historical multimodal inference logs
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    input_type VARCHAR(20) NOT NULL,            -- 'text', 'image', or 'multimodal'
    prediction VARCHAR(20) NOT NULL,            -- 'REAL', 'FAKE', or 'UNCERTAIN'
    confidence FLOAT NOT NULL,                  -- Confidence score (0.0 - 100.0%)
    sentiment VARCHAR(50),                      -- Sentiment label + emoji (e.g. 'Neutral 😐', 'Negative 😠')
    input_text TEXT,                            -- The claim / headline analyzed
    xai_summary TEXT,                           -- JSON serialized forensic breakdown & SHAP/keyword details
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 3. Notifications Table: In-app user notifications and alerts
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title VARCHAR(150) NOT NULL,
    message TEXT NOT NULL,
    type VARCHAR(50) DEFAULT 'info' NOT NULL,   -- 'info', 'success', 'warning', 'danger'
    is_read BOOLEAN NOT NULL DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 4. Password Resets Table: Cryptographic reset tokens for account recovery
CREATE TABLE IF NOT EXISTS password_resets (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token_hash VARCHAR(64) NOT NULL,            -- SHA-256 hash of secrets.token_urlsafe(32)
    expires_at DATETIME NOT NULL,               -- 30-minute validity expiry timestamp
    used BOOLEAN NOT NULL DEFAULT 0,            -- Invalidation flag upon consumption
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Indexes for Query Performance
CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_password_resets_token ON password_resets(token_hash);

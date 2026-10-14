# DeepGuard Database Architecture

This directory defines the relational database architecture and schemas for the DeepGuard Multimodal Detection System.

## Database Engine
- **Engine:** SQLite 3
- **ORM:** SQLAlchemy via Flask-SQLAlchemy
- **Active Instance Path:** `backend/instance/deepguard.db`

---

## Schema Overview

The database contains 4 relational tables:

```mermaid
erDiagram
    USERS ||--o{ PREDICTIONS : creates
    USERS ||--o{ NOTIFICATIONS : receives
    USERS ||--o{ PASSWORD_RESETS : requests

    USERS {
        int id PK
        string name
        string email UK
        string password_hash
        datetime created_at
    }

    PREDICTIONS {
        int id PK
        int user_id FK
        string input_type
        string prediction
        float confidence
        string sentiment
        text input_text
        text xai_summary
        datetime created_at
    }

    NOTIFICATIONS {
        int id PK
        int user_id FK
        string title
        text message
        string type
        bool is_read
        datetime created_at
    }

    PASSWORD_RESETS {
        int id PK
        int user_id FK
        string token_hash
        datetime expires_at
        bool used
        datetime created_at
    }
```

---

## Files in this Directory

| File | Purpose |
|---|---|
| [`schema.sql`](schema.sql) | Complete DDL script for table creation, constraints, and indexes. |
| [`db_status.py`](db_status.py) | Python utility to check table row counts and verify database integrity. |

## Quick Health Check
Run the inspection utility from anywhere:
```bash
python database/db_status.py
```

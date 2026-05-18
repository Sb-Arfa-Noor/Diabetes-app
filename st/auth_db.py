import sqlite3
import hashlib

DB_NAME = "app.db"

# ================= CREATE TABLE =================
def create_users_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT
    )
    """)

    conn.commit()
    conn.close()


# ================= PASSWORD HASH =================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ================= ADD USER =================
def add_user(username, password, role):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    try:
        c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hash_password(password), role)
        )
        conn.commit()
    except:
        pass

    conn.close()


# ================= LOGIN =================
def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute(
        "SELECT role FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    )

    result = c.fetchone()
    conn.close()

    if result:
        return result[0]
    return None
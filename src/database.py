import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("NEONDB_URL")
SQL_DIR = os.path.join(os.path.dirname(__file__), '..', 'sql', 'queries')

if not DATABASE_URL:
    raise ValueError("NEONDB_URL environment variable is not set")

# Clean up the DATABASE_URL - remove any extra quotes
DATABASE_URL = DATABASE_URL.strip().strip("'").strip('"')


def get_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception:
        raise


@contextmanager
def get_db_cursor(commit=True):
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        yield cursor
        if commit:
            conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def test_connection():
    try:
        with get_db_cursor(commit=False) as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            return result is not None
    except Exception:
        return False


@lru_cache(maxsize=100)
def load_sql(category: str, name: str) -> str:
    path = os.path.join(SQL_DIR, category, f"{name}.sql")
    with open(path, 'r') as f:
        return f.read()

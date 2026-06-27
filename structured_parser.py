import pandas as pd
import json
import xml.etree.ElementTree as ET
import sqlite3
import os
import re

SQLITE_DB_PATH = "./structured_data.db"


def get_db_connection():
    return sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)


def sanitize_table_name(filename: str) -> str:
    name = os.path.splitext(filename)[0].lower()
    name = re.sub(r'[^a-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')

    if not name or name[0].isdigit():
        name = 'tbl_' + name

    return name


def sanitize_column_name(col: str) -> str:
    """
    Same sanitization as before, but pulled into its own function
    so it can be reused consistently and tested in isolation.
    """
    clean = re.sub(r'[^a-z0-9_]', '_', str(col).lower().strip())
    clean = re.sub(r'_+', '_', clean).strip('_')
    if not clean:
        clean = "column"
    if clean[0].isdigit():
        clean = "col_" + clean
    return clean


def save_column_mapping(conn: sqlite3.Connection, table_name: str, original_columns: list, sanitized_columns: list):
    """
    Stores original → sanitized column name pairs in a dedicated
    mapping table. This is what lets the SQL router translate a
    user's natural-language question (which uses real column names
    like 'Total Revenue ($)') into the actual sanitized SQL column
    ('total_revenue') — without this, the LLM has to guess.
    """
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS _column_map (
            table_name TEXT,
            original_name TEXT,
            sanitized_name TEXT
        )
    """)
    # Clear any existing mapping for this table (re-upload case)
    cursor.execute("DELETE FROM _column_map WHERE table_name = ?", (table_name,))

    rows = [(table_name, orig, san) for orig, san in zip(original_columns, sanitized_columns)]
    cursor.executemany(
        "INSERT INTO _column_map (table_name, original_name, sanitized_name) VALUES (?, ?, ?)",
        rows
    )
    conn.commit()


def get_column_mapping(conn: sqlite3.Connection, table_name: str) -> dict:
    """
    Returns {sanitized_name: original_name} for a given table.
    Used when building the schema description for the LLM, so it
    sees the REAL column names a user would type, not the mangled ones.
    """
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT sanitized_name, original_name FROM _column_map WHERE table_name = ?",
            (table_name,)
        )
        return dict(cursor.fetchall())
    except sqlite3.OperationalError:
        # _column_map doesn't exist yet — no structured files uploaded ever
        return {}


def get_table_schema(conn: sqlite3.Connection) -> dict:
    """
    Returns all table names and their SANITIZED column names —
    unchanged from before. Use get_column_mapping() separately
    to recover original names when building prompts.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != '_column_map'")
    tables = cursor.fetchall()

    schema = {}
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        schema[table_name] = columns

    return schema


def get_full_schema_with_originals(conn: sqlite3.Connection) -> dict:
    """
    New helper: returns schema where each column is shown as
    'sanitized_name (originally: "Original Name")' — this is what
    actually gets handed to the LLM so it can match a user's
    question wording to the real SQL column.
    """
    schema = get_table_schema(conn)
    full_schema = {}

    for table_name, columns in schema.items():
        col_map = get_column_mapping(conn, table_name)
        described_columns = []
        for col in columns:
            original = col_map.get(col)
            if original and original.lower() != col.lower():
                described_columns.append(f'{col} (originally "{original}")')
            else:
                described_columns.append(col)
        full_schema[table_name] = described_columns

    return full_schema


# ── Parsers per file type ─────────────────────────────────────────────────

def parse_excel(file_path: str, filename: str) -> dict:
    conn = get_db_connection()
    tables_created = []

    try:
        xl = pd.ExcelFile(file_path)

        for sheet_name in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

            original_columns = [str(c) for c in df.columns]
            sanitized_columns = [sanitize_column_name(c) for c in original_columns]
            df.columns = sanitized_columns

            base = sanitize_table_name(filename)
            sheet_safe = re.sub(r'[^a-z0-9_]', '_', sheet_name.lower())
            table_name = f"{base}_{sheet_safe}"

            df.to_sql(table_name, conn, if_exists='replace', index=False)
            save_column_mapping(conn, table_name, original_columns, sanitized_columns)

            tables_created.append({
                "table": table_name,
                "rows": len(df),
                "columns": sanitized_columns,
                "original_columns": original_columns
            })

    finally:
        conn.close()

    return {
        "filename": filename,
        "type": "excel",
        "tables_created": tables_created
    }


def parse_csv(file_path: str, filename: str) -> dict:
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1')

    original_columns = [str(c) for c in df.columns]
    sanitized_columns = [sanitize_column_name(c) for c in original_columns]
    df.columns = sanitized_columns

    table_name = sanitize_table_name(filename)

    conn = get_db_connection()
    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        save_column_mapping(conn, table_name, original_columns, sanitized_columns)
    finally:
        conn.close()

    return {
        "filename": filename,
        "type": "csv",
        "tables_created": [{
            "table": table_name,
            "rows": len(df),
            "columns": sanitized_columns,
            "original_columns": original_columns
        }]
    }


def parse_json(file_path: str, filename: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    df = pd.json_normalize(data)

    original_columns = [str(c) for c in df.columns]
    sanitized_columns = [sanitize_column_name(c) for c in original_columns]
    df.columns = sanitized_columns

    table_name = sanitize_table_name(filename)

    conn = get_db_connection()
    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        save_column_mapping(conn, table_name, original_columns, sanitized_columns)
    finally:
        conn.close()

    return {
        "filename": filename,
        "type": "json",
        "tables_created": [{
            "table": table_name,
            "rows": len(df),
            "columns": sanitized_columns,
            "original_columns": original_columns
        }]
    }


def parse_xml(file_path: str, filename: str) -> dict:
    tree = ET.parse(file_path)
    root = tree.getroot()

    rows = []
    for child in root:
        row = {}
        for subelem in child:
            row[subelem.tag] = subelem.text
        row.update(child.attrib)
        if row:
            rows.append(row)

    if not rows:
        return {"filename": filename, "type": "xml",
                "error": "No parseable rows found in XML"}

    df = pd.DataFrame(rows)

    original_columns = [str(c) for c in df.columns]
    sanitized_columns = [sanitize_column_name(c) for c in original_columns]
    df.columns = sanitized_columns

    table_name = sanitize_table_name(filename)

    conn = get_db_connection()
    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        save_column_mapping(conn, table_name, original_columns, sanitized_columns)
    finally:
        conn.close()

    return {
        "filename": filename,
        "type": "xml",
        "tables_created": [{
            "table": table_name,
            "rows": len(df),
            "columns": sanitized_columns,
            "original_columns": original_columns
        }]
    }
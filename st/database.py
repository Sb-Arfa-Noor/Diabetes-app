import sqlite3
from datetime import datetime

# ================= DATABASE NAME =================
DB_NAME = "patients.db"


# ================= CONNECTION =================
def connect():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


# ================= CREATE TABLE =================
def create_table():
    conn = connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        contact TEXT,
        stage TEXT,
        type TEXT,
        bmi REAL,
        fbs REAL,
        hba1c REAL,
        ldl REAL,
        hdl REAL,
        date TEXT
    )
    """)

    conn.commit()
    conn.close()


# ================= ADD PATIENT =================
def add_patient(patient):
    conn = connect()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO patients (
            name, age, gender, contact,
            stage, type, bmi, fbs, hba1c,
            ldl, hdl, date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient.get("Name"),
        patient.get("Age"),
        patient.get("Gender"),
        patient.get("Contact"),
        patient.get("Stage"),
        patient.get("Type"),
        patient.get("BMI"),
        patient.get("FBS"),
        patient.get("HbA1c"),
        patient.get("LDL"),
        patient.get("HDL"),
        datetime.now().strftime("%Y-%m-%d")
    ))

    conn.commit()
    conn.close()


# ================= GET ALL PATIENTS =================
def get_patients():
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT * FROM patients ORDER BY id DESC")
    rows = cur.fetchall()

    conn.close()

    return [
        {
            "ID": r[0],
            "Name": r[1],
            "Age": r[2],
            "Gender": r[3],
            "Contact": r[4],
            "Stage": r[5],
            "Type": r[6],
            "BMI": r[7],
            "FBS": r[8],
            "HbA1c": r[9],
            "LDL": r[10],
            "HDL": r[11],
            "Date": r[12],
        }
        for r in rows
    ]


# ================= DELETE SINGLE PATIENT =================
def delete_patient(pid):
    conn = connect()
    cur = conn.cursor()

    cur.execute("DELETE FROM patients WHERE id=?", (pid,))

    conn.commit()
    conn.close()


# ================= DELETE MULTIPLE PATIENTS =================
def delete_multiple(ids):
    conn = connect()
    cur = conn.cursor()

    cur.executemany(
        "DELETE FROM patients WHERE id=?",
        [(i,) for i in ids]
    )

    conn.commit()
    conn.close()


# ================= GET SINGLE PATIENT =================
def get_patient(pid):
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT * FROM patients WHERE id=?", (pid,))
    row = cur.fetchone()

    conn.close()

    if row:
        return {
            "ID": row[0],
            "Name": row[1],
            "Age": row[2],
            "Gender": row[3],
            "Contact": row[4],
            "Stage": row[5],
            "Type": row[6],
            "BMI": row[7],
            "FBS": row[8],
            "HbA1c": row[9],
            "LDL": row[10],
            "HDL": row[11],
            "Date": row[12],
        }

    return None
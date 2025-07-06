import sqlite3
from datetime import datetime

def get_db_connection():
    conn = sqlite3.connect('attendance.db')
    return conn
def mark_attendance(name):
    print(name)
    if name == "Unknown":
        return  # Ignore unknown faces

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Ensure the attendance table exists
        cur.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Check if attendance is already marked for today
        cur.execute("""
            SELECT * FROM attendance 
            WHERE student_name = ? 
            AND DATE(timestamp) = DATE('now');
        """, (name,))
        
        result = cur.fetchone()
        
        if result is None:  # If no record exists, mark attendance
            cur.execute("INSERT INTO attendance (student_name) VALUES (?)", (name,))
            conn.commit()
            print(f"✅ Attendance marked for {name}")
        else:
            print(f"⏳ Attendance already marked today for {name}")

        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error marking attendance: {e}")

def view_attendance():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT student_name, timestamp 
            FROM attendance 
            WHERE DATE(timestamp) = DATE('now') AND marked=true
            ORDER BY timestamp DESC;
        """)
        attendance_records = cur.fetchall()

        cur.close()
        conn.close()
        return attendance_records
    except Exception as e:
        print(f"❌ Error fetching attendance: {e}")


def review_attendance():
    try:
        conn = sqlite3.connect("attendance.db")
        cur = conn.cursor()

        cur.execute("""
            SELECT id, student_name,marked, timestamp 
            FROM attendance 
            WHERE DATE(timestamp) = DATE('now') 
            ORDER BY timestamp DESC;
        """)
        attendance_records = cur.fetchall()

        cur.close()
        conn.close()
        return attendance_records
    except Exception as e:
        print(f"❌ Error fetching attendance: {e}")

def update_attendance(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE attendance SET marked = ? WHERE id = ? AND marked = ?", (int(data['marked']), data['id'],data['timestamp']))
    conn.commit()
    conn.close()
def get_attendance():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, student_name, marked, timestamp FROM attendance")
    data = cursor.fetchall()
    conn.close()
    return data
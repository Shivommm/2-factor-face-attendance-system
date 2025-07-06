import sqlite3

# Connect to a database file (creates if it doesn't exist)
conn = sqlite3.connect("attendance.db")

# Create a cursor object
cursor = conn.cursor()

# Check SQLite version
cursor.execute("DELETE FROM attendance;")
conn.commit()

# Close the connection
conn.close()

import psycopg2
try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="yourpassword",  # Set your actual PostgreSQL password
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()
        # cur.execute("TRUNCATE attendance;")
        cur.execute("SELECT * FROM attendance;")
        result = cur.fetchall()
        print(result)
        # conn.commit()
        cur.close()
        conn.close()
except Exception as e :
    print(f'err {e}')
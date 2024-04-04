import sqlite3

DATABASE_FILE = 'chatbot.db'


def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        print(f"Connected to {DATABASE_FILE}")
    except sqlite3.Error as e:
        print(e)

    return conn


def init_db():
    conn = create_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS conversations
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, response TEXT)''')
            conn.commit()
            print("Database initialized successfully")
        except sqlite3.Error as e:
            print(e)
        finally:
            conn.close()


def store_conversation(question, response):
    conn = create_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute("INSERT INTO conversations (question, response) VALUES (?, ?)", (question, response))
            conn.commit()
            print("Conversation stored successfully")
        except sqlite3.Error as e:
            print(e)
        finally:
            conn.close()

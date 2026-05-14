import sqlite3
import json
import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "experiments.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        parameters TEXT NOT NULL,
        metrics TEXT NOT NULL,
        artifacts TEXT NOT NULL,
        notes TEXT
    )''')
    conn.commit()
    conn.close()

def log_experiment(project, parameters, metrics, artifacts, notes=None):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO experiments 
        (project, timestamp, parameters, metrics, artifacts, notes)
        VALUES (?, ?, ?, ?, ?, ?)''',
        (project, datetime.datetime.now().isoformat(),
         json.dumps(parameters), json.dumps(metrics),
         json.dumps(artifacts), notes))
    conn.commit()
    conn.close()
    print(f"\n[Tracking] Experimento registrado soberanamente en {DB_PATH}")

import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "experiments.db")

def list_experiments(project=None, limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if project:
        c.execute("SELECT * FROM experiments WHERE project=? ORDER BY timestamp DESC LIMIT ?", (project, limit))
    else:
        c.execute("SELECT * FROM experiments ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    
    for row in rows:
        print(f"\n[{row[1]}] ID: {row[0]} @ {row[2]}")
        print(f"  Parámetros: {json.loads(row[3])}")
        print(f"  Métricas: {json.loads(row[4])}")
        print(f"  Artefactos: {json.loads(row[5])}")
        if row[6]: print(f"  Notas: {row[6]}")
    
    return rows

def compare_metric(project, metric_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, metrics FROM experiments WHERE project=? ORDER BY timestamp ASC", (project,))
    rows = c.fetchall()
    conn.close()
    
    timestamps = []
    values = []
    for row in rows:
        metrics = json.loads(row[1])
        if metric_name in metrics:
            # Simplificar timestamp para el plot (ej. 'YYYY-MM-DD HH:MM')
            ts = row[0][:16].replace('T', ' ')
            timestamps.append(ts)
            values.append(metrics[metric_name])
    
    return timestamps, values

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query Quantum Lab Experiments")
    parser.add_argument("--project", "-p", type=str, help="Filtrar por proyecto (ej. P3_G2)")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Número máximo de resultados")
    args = parser.parse_args()
    
    if not os.path.exists(DB_PATH):
        print(f"Error: No existe la base de datos {DB_PATH}")
    else:
        list_experiments(args.project, args.limit)

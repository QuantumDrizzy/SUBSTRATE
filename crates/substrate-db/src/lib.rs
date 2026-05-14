use anyhow::Result;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

pub struct SubstrateDb {
    conn: Connection,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunRecord {
    pub id:            i64,
    pub timestamp:     String,
    pub layer:         String,
    pub score:         f64,
    pub metadata_json: String,
}

impl SubstrateDb {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        let db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS runs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                layer         TEXT    NOT NULL,
                score         REAL    NOT NULL DEFAULT 0.0,
                metadata_json TEXT    NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_runs_layer ON runs(layer);

            CREATE TABLE IF NOT EXISTS config (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );",
        )?;
        Ok(())
    }

    pub fn store_run(
        &self,
        layer: &str,
        score: f64,
        metadata: &serde_json::Value,
    ) -> Result<i64> {
        let ts   = chrono::Utc::now().to_rfc3339();
        let meta = serde_json::to_string(metadata)?;
        self.conn.execute(
            "INSERT INTO runs (timestamp, layer, score, metadata_json) VALUES (?1,?2,?3,?4)",
            params![ts, layer, score, meta],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Most recent run per layer.
    pub fn get_latest(&self) -> Result<Vec<RunRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, layer, score, metadata_json
             FROM runs
             WHERE id IN (SELECT MAX(id) FROM runs GROUP BY layer)
             ORDER BY layer",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(RunRecord {
                id:            row.get(0)?,
                timestamp:     row.get(1)?,
                layer:         row.get(2)?,
                score:         row.get(3)?,
                metadata_json: row.get(4)?,
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Most recent `n` run rows, newest first.
    pub fn get_history(&self, n: usize) -> Result<Vec<RunRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, layer, score, metadata_json
             FROM runs ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![n as i64], |row| {
            Ok(RunRecord {
                id:            row.get(0)?,
                timestamp:     row.get(1)?,
                layer:         row.get(2)?,
                score:         row.get(3)?,
                metadata_json: row.get(4)?,
            })
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    pub fn set_config(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?1,?2)",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn get_config(&self, key: &str) -> Result<Option<String>> {
        match self.conn.query_row(
            "SELECT value FROM config WHERE key = ?1",
            params![key],
            |row| row.get(0),
        ) {
            Ok(v)                                        => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows)   => Ok(None),
            Err(e)                                       => Err(e.into()),
        }
    }
}

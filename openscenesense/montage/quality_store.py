"""Durable attempt ledger. An interrupted request may have been billed upstream."""

from __future__ import annotations

import json
import sqlite3
import threading
import time


class BudgetReached(RuntimeError):
    pass


def tokens(usage):
    input_count = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_count = usage.get("output_tokens", usage.get("completion_tokens"))
    if any(type(x) is not int or x < 0 for x in (input_count, output_count)):
        return None
    details = usage.get("input_tokens_details") or {}
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    if type(cached) is not int or not 0 <= cached <= input_count:
        cached = 0
    return input_count, output_count, cached


class Store:
    def __init__(self, path, run_key, max_calls, max_tokens):
        self.guard = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
              id INTEGER PRIMARY KEY, task TEXT NOT NULL, started REAL NOT NULL,
              finished REAL, status TEXT NOT NULL, reservation INTEGER NOT NULL,
              usage TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS results (task TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        previous = self.db.execute("SELECT value FROM metadata WHERE key='run'").fetchone()
        if previous and previous[0] != run_key:
            self.db.close()
            raise ValueError(
                "Different model/prompt/transcript in this run: choose a new directory"
            )
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('run', ?)", (run_key,))
        self.db.execute(
            "UPDATE attempts SET status='interrupted', "
            "error='Completion unknown after interruption' WHERE status='running'"
        )
        self.db.commit()
        self.max_calls, self.max_tokens = max_calls, max_tokens

    def cached(self, task):
        with self.guard:
            row = self.db.execute("SELECT value FROM results WHERE task=?", (task,)).fetchone()
            return json.loads(row[0]) if row else None

    def _summary(self):
        rows = self.db.execute("SELECT status,reservation,usage FROM attempts").fetchall()
        inp = out = cached = unknown = reserved = 0
        for status, reservation, usage in rows:
            counts = tokens(json.loads(usage)) if usage else None
            if counts is not None:
                a, b, c = counts
                inp += a
                out += b
                cached += c
            else:
                reserved += reservation
                if status != "running":
                    unknown += 1
        return {
            "attempts": len(rows),
            "input_tokens": inp,
            "output_tokens": out,
            "cached_input_tokens": cached,
            "unknown_usage_attempts": unknown,
            "reserved_or_unknown_tokens": reserved,
            "accounted_tokens": inp + out + reserved,
        }

    def summary(self):
        with self.guard:
            return self._summary()

    def begin(self, task, reservation):
        with self.guard:
            summary = self._summary()
            if summary["attempts"] >= self.max_calls:
                raise BudgetReached("Cumulative --max-calls reached; raise it explicitly to resume")
            if summary["accounted_tokens"] + reservation > self.max_tokens:
                raise BudgetReached(
                    "Cumulative --max-tokens reservation reached; raise it explicitly to resume"
                )
            cursor = self.db.execute(
                "INSERT INTO attempts(task,started,status,reservation) VALUES (?,?,'running',?)",
                (task, time.time(), reservation),
            )
            self.db.commit()
            return cursor.lastrowid

    def finish(self, attempt, task, value, usage, error=None):
        with self.guard:
            self.db.execute(
                "UPDATE attempts SET finished=?,status=?,usage=?,error=? WHERE id=?",
                (time.time(), "failed" if error else "complete", json.dumps(usage), error, attempt),
            )
            if value is not None and error is None:
                self.db.execute(
                    "INSERT OR REPLACE INTO results VALUES (?,?)",
                    (task, json.dumps(value, ensure_ascii=False)),
                )
            self.db.commit()

    def close(self):
        self.db.close()

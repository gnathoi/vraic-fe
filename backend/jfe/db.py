"""PostgreSQL is the authoritative store for sessions, drafts, runs, results and saved items.
The run queue is a table: the worker claims work with FOR UPDATE SKIP LOCKED (no separate broker)."""
import os
import secrets

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

SCHEMA = """
create table if not exists sessions (
  id text primary key, created timestamptz not null default now(), expires timestamptz not null,
  is_operator boolean not null default false
);
create table if not exists drafts (
  id text not null, session_id text not null references sessions(id), version int not null,
  status text not null, scenario jsonb, parent_run_id text, message text, detail jsonb,
  created timestamptz not null default now(), primary key (id, version)
);
do $$ begin
  if (select count(*) from information_schema.key_column_usage where table_name = 'drafts' and constraint_name = 'drafts_pkey') = 1 then
    alter table drafts drop constraint drafts_pkey;
    alter table drafts add primary key (id, version);
  end if;
end $$;
create table if not exists results (
  experiment_hash text primary key, result jsonb not null, created timestamptz not null default now()
);
create table if not exists runs (
  id text primary key, session_id text not null references sessions(id), draft_id text, draft_version int,
  scenario jsonb not null, experiment_hash text not null, status text not null, is_operator boolean not null,
  idempotency_key text, created timestamptz not null default now(), started timestamptz, finished timestamptz,
  lease_until timestamptz, attempts int not null default 0, error text, cache_status text, backend text,
  queue_seconds real, compute_seconds real, artifacts_status text, seq bigserial,
  unique (session_id, idempotency_key)
);
create index if not exists runs_queue on runs (status, is_operator, created);
create table if not exists saved (
  id text primary key, session_id text not null references sessions(id), run_id text not null references runs(id),
  name text not null, note text, created timestamptz not null default now()
);
create table if not exists settings (key text primary key, value text not null);
create table if not exists events (
  id bigserial primary key, at timestamptz not null default now(), kind text not null, status text, latency real
);
create index if not exists events_at on events (at);
create table if not exists heartbeat (component text primary key, at timestamptz not null default now(), data jsonb not null);
insert into settings values ('mode', 'OPEN_DEMO') on conflict do nothing;
"""

_pool = None


def pool():
    global _pool
    if _pool is None:
        _pool = ConnectionPool(os.environ.get("JFE_DATABASE_URL", "postgresql://jfe:jfe@localhost:5432/jfe"),
                               min_size=1, max_size=int(os.environ.get("JFE_DB_POOL", "10")),
                               kwargs={"row_factory": dict_row, "autocommit": True}, open=True)
    return _pool


def init():
    with pool().connection() as c:
        c.execute("select pg_advisory_lock(42)")
        try:
            c.execute(SCHEMA)
        finally:
            c.execute("select pg_advisory_unlock(42)")


def q(sql, *args):
    with pool().connection() as c:
        cur = c.execute(sql, args)
        return cur.fetchall() if cur.description else []


def one(sql, *args):
    rows = q(sql, *args)
    return rows[0] if rows else None


def new_id(prefix):
    return f"{prefix}_{secrets.token_urlsafe(9)}"


def event(kind, status=None, latency=None):
    """Operational telemetry only: no prompt text, titles or identifiers."""
    q("insert into events (kind, status, latency) values (%s, %s, %s)", kind, status, latency)


def mode():
    return one("select value from settings where key='mode'")["value"]


J = Jsonb

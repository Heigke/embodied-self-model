#!/usr/bin/env python3
"""
Resumable checkpointing for long (night-long) parameter sweeps.

The container is ephemeral and can be reclaimed on inactivity, so every cell is
flushed to a JSONL the instant it finishes. On restart the sweep skips cells
already present (keyed by a stable cell-id) and continues. A compact summary
JSON is rewritten periodically and is the artifact worth committing.

Model-agnostic: run_fn(params) -> dict of scalar results. Nothing here knows
about the physics; the advanced model plugs in via run_fn.
"""
import json, os, hashlib, time


def cell_id(params):
    """Stable id for a parameter dict (order-independent)."""
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _load_done(jsonl_path):
    done = {}
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[rec["_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a torn last line from a hard kill
    return done


def run_grid(cells, run_fn, jsonl_path, summary_path=None,
             summary_every=25, clock=None, on_summary=None):
    """
    cells       : list of param dicts to evaluate
    run_fn      : params -> dict of scalar results (must be JSON-serialisable)
    jsonl_path  : append-only per-cell log; the resume source of truth
    summary_path: compact JSON rewritten every `summary_every` cells (committable)
    clock       : callable -> float seconds (injected; Date.now is unavailable in
                  some contexts, so the caller passes time.time)
    on_summary  : optional callback(list_of_records) run at each summary flush
    Returns the full list of records (done + newly computed).
    """
    clock = clock or time.time
    done = _load_done(jsonl_path)
    total = len(cells)
    t0 = clock()
    with open(jsonl_path, "a") as fh:
        for i, params in enumerate(cells):
            cid = cell_id(params)
            if cid in done:
                continue
            t_cell = clock()
            try:
                res = run_fn(params)
                status = "ok"
            except Exception as e:                      # never let one cell kill the night
                res = {"error": repr(e)}
                status = "error"
            rec = {"_id": cid, "_status": status, "_secs": round(clock() - t_cell, 3),
                   **params, **res}
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())                        # survive a hard kill
            done[cid] = rec
            n = len(done)
            if n % 5 == 0 or n == total:
                el = clock() - t0
                rate = n / el if el > 0 else 0
                eta = (total - n) / rate if rate > 0 else float("inf")
                print(f"  [{n:>5}/{total}]  {status:5s}  cell {res.get('_tag','')}"
                      f"  elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
            if summary_path and (n % summary_every == 0 or n == total):
                _write_summary(summary_path, list(done.values()))
                if on_summary:
                    on_summary(list(done.values()))
    if summary_path:
        _write_summary(summary_path, list(done.values()))
        if on_summary:
            on_summary(list(done.values()))
    return list(done.values())


def _write_summary(path, records):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"n": len(records), "records": records}, f, default=str)
    os.replace(tmp, path)   # atomic; a reclaim mid-write never corrupts the summary

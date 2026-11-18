#!/usr/bin/env python3
"""Probe an Iron Fish stratum v1 endpoint.

Usage:
    python tools/probe_stratum.py HOST PORT [--worker NAME] [--listen SEC]

What it does:
    1. TCP-connects to HOST:PORT.
    2. Waits for a server push (some pools send the first job).
    3. Sends a stratum v1 `subscribe`.
    4. Prints every raw message for --listen seconds.

Goal: learn the pool's exact message shapes (job fields, notify shape)
so the miner client can parse them. No mining happens.
"""
import argparse
import json
import socket
import sys
import time


def recv_lines(sock, deadline):
    """Yield decoded lines until the deadline or EOF."""
    buf = b""
    while time.time() < deadline:
        sock.settimeout(max(0.1, deadline - time.time()))
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            continue
        except (ConnectionResetError, TimeoutError):
            print("!! connection reset/timeout mid-read", flush=True)
            return
        if not chunk:
            print("!! EOF", flush=True)
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode("utf-8", "replace")
    if buf.strip():
        print(f"!! partial line at deadline: {buf.decode('utf-8', 'replace')!r}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("host")
    ap.add_argument("port", type=int)
    ap.add_argument("--worker", default="fishpoc@probe",
                    help="username for subscribe (pool may reject a fake address)")
    ap.add_argument("--listen", type=float, default=8.0,
                    help="seconds to collect messages after subscribe")
    ap.add_argument("--push-wait", type=float, default=2.0,
                    help="seconds to wait for an initial server push")
    args = ap.parse_args()

    print(f"== probe {args.host}:{args.port} ==", flush=True)
    try:
        sock = socket.create_connection((args.host, args.port), timeout=10)
    except OSError as e:
        print(f"!! connect failed: {e}", flush=True)
        sys.exit(2)

    for line in recv_lines(sock, time.time() + args.push_wait):
        print(f"[push] {line}", flush=True)

    sub = {"id": 1, "method": "subscribe", "params": [args.worker, []]}
    payload = json.dumps(sub) + "\n"
    sock.sendall(payload.encode())
    print(f"[sent] {payload.strip()}", flush=True)

    for line in recv_lines(sock, time.time() + args.listen):
        print(f"[recv] {line}", flush=True)

    sock.close()
    print("== done ==", flush=True)


if __name__ == "__main__":
    main()

"""Launches both the FastAPI backend and the Streamlit frontend for local
development, so the web app can be started with one command instead of two
terminals. See README's "Web app" section to run them separately.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
API_PORT = os.environ.get("NBA_AGENT_API_PORT", "8000")


def _raise_keyboard_interrupt(signum, frame):
    # Treat SIGTERM (e.g. from a process manager, or `kill` without -INT)
    # the same as Ctrl+C, so both paths hit the same cleanup below instead
    # of leaving the child processes orphaned.
    raise KeyboardInterrupt


def main() -> None:
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    # No --reload here: its reloader+worker multiprocessing setup doesn't
    # always die cleanly on terminate(), which can hang this script's own
    # shutdown. Use `uvicorn nba_live_agent.api:app --reload` directly (see
    # README) if you want hot-reload during backend-focused development.
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "nba_live_agent.api:app", "--port", API_PORT],
        env=env,
    )
    streamlit_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py"],
        env=env,
    )
    procs = [api_proc, streamlit_proc]

    try:
        while True:
            for proc in procs:
                code = proc.poll()
                if code is not None:
                    print(f"\n{proc.args[2]} exited with code {code} — shutting down.")
                    raise SystemExit(1)
            time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()

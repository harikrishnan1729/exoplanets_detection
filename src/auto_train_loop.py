"""
Auto-training loop for exoplanet detector.

Workflow:
  1. Run evaluate.py, capture its stdout, parse the line "f1: <score>".
  2. Compare against the best F1 seen so far (stored in best_f1.txt).
  3. If it's a NEW BEST:
       - save it
       - git add / commit (message includes the score) / push
       - stop (or keep going, see CONTINUE_AFTER_BEST below)
  4. If it's NOT better:
       - run train.py, wait for it to finish
       - run evaluate.py again
       - repeat until F1 "saturates" (stops improving meaningfully)

Run this from anywhere - it uses absolute paths below. Adjust PROJECT_DIR
if you move the project.
"""

import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG - edit these if your paths change
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(r"c:\Users\HARIKRISHNAN\Desktop\exoplanet dectetor")
PYTHON_EXE = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
EVALUATE_SCRIPT = PROJECT_DIR / "src" / "evaluate.py"
TRAIN_SCRIPT = PROJECT_DIR / "src" / "train.py"

BEST_F1_FILE = PROJECT_DIR / "best_f1.txt"
LOG_FILE = PROJECT_DIR / "auto_train_log.txt"

# Regex to find "f1: 0.8123" (case-insensitive, tolerant of spacing)
F1_PATTERN = re.compile(r"f1\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# Saturation controls
MAX_ITERATIONS = 50          # hard safety cap on train/eval cycles
PATIENCE = 4                 # how many non-improving rounds in a row before we stop
MIN_IMPROVEMENT = 0.001      # improvement smaller than this doesn't reset patience

# If True: after pushing a new best, keep looping to try to beat it again.
# If False: stop as soon as one new best is committed & pushed.
CONTINUE_AFTER_BEST = True


# ---------------------------------------------------------------------------
def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_and_capture(cmd, cwd):
    """Run a command, stream its output live, and return the full captured text."""
    log(f"Running: {' '.join(str(c) for c in cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured_lines = []
    for line in process.stdout:
        print(line, end="")  # live output so you can watch training progress
        captured_lines.append(line)
    process.wait()
    if process.returncode != 0:
        log(f"WARNING: command exited with code {process.returncode}")
    return "".join(captured_lines)


def parse_f1(output: str):
    matches = F1_PATTERN.findall(output)
    if not matches:
        return None
    # Use the LAST match in case the script prints intermediate f1 values too
    return float(matches[-1])


def get_best_f1():
    if BEST_F1_FILE.exists():
        try:
            return float(BEST_F1_FILE.read_text().strip())
        except ValueError:
            return -1.0
    return -1.0


def save_best_f1(value: float):
    BEST_F1_FILE.write_text(f"{value:.6f}")


def git(cmd_args, cwd):
    result = subprocess.run(
        ["git"] + cmd_args, cwd=cwd, capture_output=True, text=True
    )
    log(f"git {' '.join(cmd_args)} -> rc={result.returncode}")
    if result.stdout.strip():
        log(result.stdout.strip())
    if result.stderr.strip():
        log(result.stderr.strip())
    return result.returncode == 0


def commit_and_push_best(f1_value: float):
    message = f"f1: {f1_value:.6f}"
    if not git(["add", "."], PROJECT_DIR):
        return False
    if not git(["commit", "-m", message], PROJECT_DIR):
        log("Nothing to commit or commit failed - continuing anyway.")
    return git(["push", "origin", "main"], PROJECT_DIR)


def run_evaluate():
    output = run_and_capture([str(PYTHON_EXE), str(EVALUATE_SCRIPT)], PROJECT_DIR)
    f1 = parse_f1(output)
    if f1 is None:
        log("ERROR: could not find an 'f1: <score>' line in evaluate.py output.")
    return f1


def run_train():
    run_and_capture([str(PYTHON_EXE), str(TRAIN_SCRIPT)], PROJECT_DIR)


# ---------------------------------------------------------------------------
def main():
    if not PYTHON_EXE.exists():
        log(f"ERROR: python.exe not found at {PYTHON_EXE}")
        sys.exit(1)
    if not EVALUATE_SCRIPT.exists():
        log(f"ERROR: evaluate.py not found at {EVALUATE_SCRIPT}")
        sys.exit(1)

    best_f1 = get_best_f1()
    log(f"Starting. Current best F1 on record: {best_f1}")

    no_improve_streak = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        log(f"--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        f1 = run_evaluate()
        if f1 is None:
            log("Aborting: evaluate.py did not report an f1 score.")
            sys.exit(1)

        log(f"Current F1 = {f1:.6f} | Best so far = {best_f1:.6f}")

        if f1 > best_f1 + MIN_IMPROVEMENT or best_f1 < 0:
            improvement = f1 - best_f1 if best_f1 >= 0 else f1
            log(f"New best F1! ({f1:.6f}, improvement {improvement:.6f})")
            best_f1 = f1
            save_best_f1(best_f1)
            no_improve_streak = 0

            pushed = commit_and_push_best(best_f1)
            if pushed:
                log(f"Committed and pushed best F1 = {best_f1:.6f}")
            else:
                log("Commit/push failed - check git status manually.")

            if not CONTINUE_AFTER_BEST:
                log("CONTINUE_AFTER_BEST is False - stopping after new best.")
                return
        else:
            no_improve_streak += 1
            log(f"No meaningful improvement. Streak = {no_improve_streak}/{PATIENCE}")

            if no_improve_streak >= PATIENCE:
                log(f"F1 has saturated (no improvement for {PATIENCE} rounds). Stopping.")
                return

        log("Running train.py to try for a better model...")
        run_train()
        log("Training finished. Re-evaluating...")

    log("Reached MAX_ITERATIONS cap without saturating or hitting a hard stop.")


if __name__ == "__main__":
    main()

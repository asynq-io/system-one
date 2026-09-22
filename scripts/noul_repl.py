"""Ask the local ONNX model yes/no questions about one state, interactively.

Usage:  uv run scripts/noul_repl.py [--onnx-dir DIR] [--model NAME] [state]

Unset options fall back to `SYSTEM_ONE_ONNX_DIR` / `SYSTEM_ONE_MODEL`. Without
`state` the first prompt reads it. `/state` swaps it, Ctrl-D quits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from system_one import ONNXConfig, SystemOne

YES_THRESHOLD = 0.5


def read_state() -> str:
    print("State (the situation to judge; end with an empty line):")
    lines = iter(input, "")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", nargs="?")
    parser.add_argument("--onnx-dir", type=Path)
    parser.add_argument("--model")
    args = parser.parse_args()
    config = ONNXConfig(**({"onnx_dir": args.onnx_dir} if args.onnx_dir else {}))
    overrides = {"model": args.model} if args.model else {}

    with SystemOne(config, **overrides) as agent:
        state = args.state or read_state()
        print("Ask yes/no questions. `/state` changes the state, Ctrl-D quits.")
        try:
            while True:
                question = input("? ").strip()
                if not question:
                    continue
                if question == "/state":
                    state = read_state()
                    continue
                answer = agent.ask(
                    state, {"q": {"type": "noul", "instructions": question}}
                )
                noul = answer.nouls["q"]
                verdict = "yes" if noul.noul >= YES_THRESHOLD else "no"
                print(
                    f"{verdict}  (p={noul.noul:.2f}, confidence={noul.confidence:.2f})"
                )
        except EOFError:
            print()


if __name__ == "__main__":
    main()

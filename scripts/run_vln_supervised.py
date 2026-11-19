"""Compatibility CLI for separately authorized one-action and five-move sessions."""
import sys
from go2_perception.vln.supervised import Hold
from go2_perception.vln.supervised_launch import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Hold as error:
        print("HOLD: " + str(error), file=sys.stderr)
        raise SystemExit(2)

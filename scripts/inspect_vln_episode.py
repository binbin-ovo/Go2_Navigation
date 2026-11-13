"""Minimal contract consumer; loads no model and no evaluation annotations."""
import argparse
import json
from go2_perception.vln.episodes import load_episode, select_history

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest')
    ap.add_argument('--decision', default=None)
    args = ap.parse_args()
    episode = load_episode(args.manifest)
    decision = args.decision or episode.data['decisions'][0]['decision_id']
    print(json.dumps(select_history(episode, decision), indent=2))

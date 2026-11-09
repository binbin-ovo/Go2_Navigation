"""Plot paired social forecasts with only observed neighbor context."""

import argparse
from go2_perception.forecasting.social_visualization import plot_social_examples

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    plot_social_examples(args.predictions, args.output)

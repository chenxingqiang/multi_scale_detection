import argparse
from train.train import train_model
from train.validate import validate_model


def main(args):
    if args.mode == "train":
        train_model()
    elif args.mode == "validate":
        validate_model()
    else:
        print("Unknown mode. Please choose 'train' or 'validate'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Scale Text Detection")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "validate"],
        help="Choose the mode: 'train' or 'validate'",
    )

    args = parser.parse_args()
    main(args)

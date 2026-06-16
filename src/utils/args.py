import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Facial Recognition System")
    parser.add_argument(
        "--action",
        required=True,
        choices=["sign-up", "sign-in"],
        help="Action to perform.",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["hybrid", "standard"],
        help="Recognition pipeline to use.",
    )
    parser.add_argument(
        "--image-path",
        required=True,
        nargs="+",
        help="One or more face image paths.",
    )
    parser.add_argument(
        "--enrollment-strategy",
        choices=["multi", "centroid"],
        default="multi",
        help="How sign-up embeddings are stored.",
    )
    parser.add_argument(
        "--full-name",
        help="Profile name. Required for sign-up.",
    )
    parser.add_argument(
        "--external-code",
        help="External profile identifier. Required for sign-up.",
    )

    args = parser.parse_args()
    if args.action == "sign-up":
        if not args.full_name:
            parser.error("--full-name is required for sign-up.")
        if not args.external_code:
            parser.error("--external-code is required for sign-up.")
    elif len(args.image_path) != 1:
        parser.error("sign-in requires exactly one --image-path.")

    return args

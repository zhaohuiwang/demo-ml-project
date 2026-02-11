#!/usr/bin/env python3

import argparse
import sys
from mlflow import MlflowClient
import mlflow


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign an alias to a specific MLflow model version."
    )

    parser.add_argument(
        "--model-name",
required=True,
        help="Registered model name in MLflow."
    )

    parser.add_argument(
        "--alias",
        required=True,
        help="Alias to assign (e.g., champion, staging, production)."
    )

    parser.add_argument(
        "--version",
        required=True,
        help="Model version number (e.g., 1, 2, 5)."
    )

    parser.add_argument(
        "--tracking-uri",
        required=False,
        help="Optional MLflow tracking URI. If not provided, uses current environment."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
        print(f"Using tracking URI: {args.tracking_uri}")

    client = MlflowClient()

    try:
        # Set alias
        client.set_registered_model_alias(
            name=args.model_name,
            alias=args.alias,
            version=args.version
        )

        print(
            f"✅ Alias '{args.alias}' assigned to version "
            f"{args.version} of '{args.model_name}'"
        )

        # Verify
        mv = client.get_model_version_by_alias(
            args.model_name,
            args.alias
        )

        print(f"🔎 Verified: @{args.alias} → version {mv.version}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

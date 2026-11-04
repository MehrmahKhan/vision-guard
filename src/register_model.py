"""Task 3.6 - register the best model and move the champion alias.

Run (replace the run id with the one you picked from the comparison):
    python -m src.register_model --run-id <RUN_ID> --alias champion
"""

import argparse

import mlflow
from mlflow.tracking import MlflowClient

from src import config

MODEL_NAME = "VisionGuard"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-path", default="model")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    if config.MLFLOW_TRACKING_URI != "":
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    model_uri = "runs:/" + args.run_id + "/" + args.artifact_path
    print("registering " + model_uri + " as " + args.model_name)

    result = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    print("created version " + str(result.version))

    client = MlflowClient()
    client.set_registered_model_alias(
        name=args.model_name, alias=args.alias, version=result.version
    )
    print("alias '" + args.alias + "' now points at version " + str(result.version))

    print("")
    print("all versions:")
    versions = client.search_model_versions("name='" + args.model_name + "'")
    for version in versions:
        print(
            "  version " + str(version.version)
            + "  run_id=" + str(version.run_id)
            + "  aliases=" + str(version.aliases)
        )


if __name__ == "__main__":
    main()

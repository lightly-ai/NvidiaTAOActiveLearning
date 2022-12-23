from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import DatasetType
from lightly.openapi_generated.swagger_client import DatasourcePurpose


def schedule_selection():

    # Create the Lightly client to connect to the API.
    client = ApiWorkflowClient(token="MY_LIGHTLY_TOKEN")

    # Create a new dataset on the Lightly Platform.
    dataset_name = "dataset-name"
    try:
        client.create_dataset(
            dataset_name=dataset_name,
            dataset_type=DatasetType.IMAGES,  # Can be DatasetType.VIDEOS when working with videos
        )
        print(f"Created dataset: https://app.lightly.ai/{client.dataset_id}")
    except ValueError:
        client.set_dataset_id_by_name(dataset_name)
        print(f"Re-using dataset: https://app.lightly.ai/{client.dataset_id}")

    # Configure the Input datasource.
    client.set_s3_config(
        resource_path="s3://bucket/input/",
        region="eu-central-1",
        access_key="S3-ACCESS-KEY",
        secret_access_key="S3-SECRET-ACCESS-KEY",
        purpose=DatasourcePurpose.INPUT,
    )
    # Configure the Lightly datasource.
    client.set_s3_config(
        resource_path="s3://bucket/lightly/",
        region="eu-central-1",
        access_key="S3-ACCESS-KEY",
        secret_access_key="S3-SECRET-ACCESS-KEY",
        purpose=DatasourcePurpose.LIGHTLY,
    )

    # Configure and schedule a run.
    client.schedule_compute_worker_run(
        worker_config={},
        selection_config={
            "n_samples": 50,
            "strategies": [
                {"input": {"type": "EMBEDDINGS"}, "strategy": {"type": "DIVERSITY"}},
                {
                    "input": {
                        "type": "SCORES",
                        "task": "minneapple",
                        "score": "uncertainty_entropy",
                    },
                    "strategy": {"type": "WEIGHTS"},
                },
            ],
        },
    )


if __name__ == "__main__":

    ...

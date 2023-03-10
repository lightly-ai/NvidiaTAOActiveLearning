import argparse

from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import DatasetType
from lightly.openapi_generated.swagger_client import DatasourcePurpose


def schedule_selection(
    dataset_name: str,
    s3_resource_path: str,
    s3_lightly_path: str,
    s3_region: str,
    s3_input_role_arn: str,
    s3_input_external_id: str,
    s3_lightly_role_arn: str,
    s3_lightly_external_id: str,
):

    # Create the Lightly client to connect to the API.
    client = ApiWorkflowClient()

    # Create a new dataset on the Lightly Platform.
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
    client.set_s3_delegated_access_config(
        resource_path=s3_resource_path,
        region=s3_region,
        role_arn=s3_input_role_arn,
        external_id=s3_input_external_id,
        purpose=DatasourcePurpose.INPUT,
    )
    # Configure the Lightly datasource.
    client.set_s3_delegated_access_config(
        resource_path=s3_lightly_path,
        region=s3_region,
        role_arn=s3_lightly_role_arn,
        external_id=s3_lightly_external_id,
        purpose=DatasourcePurpose.LIGHTLY,
    )

    # Configure and schedule a run.
    scheduled_run_id = client.schedule_compute_worker_run(
        worker_config={"datasource": {"process_all": True}},
        selection_config={
            "n_samples": 100,
            "strategies": [
                # Diversity
                {"input": {"type": "EMBEDDINGS"}, "strategy": {"type": "DIVERSITY"}},
                # Active learning
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

    # Watch the job
    for run_info in client.compute_worker_run_info_generator(
        scheduled_run_id=scheduled_run_id
    ):
        print(
            f"Lightly Worker run is now in state='{run_info.state}' with message='{run_info.message}'"
        )

    if run_info.ended_successfully():
        print("Success!")
    else:
        print("Failure!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--s3-input-path", type=str)
    parser.add_argument("--s3-lightly-path", type=str)
    parser.add_argument("--s3-region", type=str)
    parser.add_argument("--s3-input-role-arn", type=str)
    parser.add_argument("--s3-input-external-id", type=str)
    parser.add_argument("--s3-lightly-role-arn", type=str)
    parser.add_argument("--s3-lightly-external-id", type=str)
    args = parser.parse_args()

    schedule_selection(
        dataset_name=args.dataset_name,
        s3_resource_path=args.s3_input_path,
        s3_lightly_path=args.s3_lightly_path,
        s3_region=args.s3_region,
        s3_input_role_arn=args.s3_input_role_arn,
        s3_input_external_id=args.s3_input_external_id,
        s3_lightly_role_arn=args.s3_lightly_role_arn,
        s3_lightly_external_id=args.s3_lightly_external_id,
    )

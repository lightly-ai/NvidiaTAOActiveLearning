import argparse
import shutil
from tqdm import tqdm
from pathlib import Path

from lightly.api import ApiWorkflowClient


def annotate_images(
    dataset_name: str,
    input_dir: Path,
) -> None:

    # Create the Lightly client to connect to the API.
    client = ApiWorkflowClient()
    client.set_dataset_id_by_name(dataset_name)

    # Get filenames of all the selected images
    tasks = client.export_label_studio_tasks_by_tag_name("initial-tag")
    filenames = [task["data"]["lightlyFileName"] for task in tasks]

    output_dir = input_dir / "train"
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Copy all images and labels from raw/ to train/
    for filename in tqdm(filenames):

        image_source = input_dir / filename
        label_source = input_dir / "raw/labels" / image_source.with_suffix(".txt").name

        image_target = input_dir / "train/images" / image_source.name
        label_target = input_dir / "train/labels" / label_source.name

        try:
            shutil.copyfile(label_source, label_target)
            shutil.copyfile(image_source, image_target)
        except FileNotFoundError:
            pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--input-dir", type=str)
    args = parser.parse_args()

    annotate_images(dataset_name=args.dataset_name, input_dir=Path(args.input_dir))

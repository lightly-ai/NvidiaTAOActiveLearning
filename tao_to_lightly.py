import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np


def tao_to_lightly(input_dir: Path) -> None:

    task_name = "minneapple"

    # Create the necessary directories.
    output_dir = Path(".lightly/")
    prediction_dir = output_dir / "predictions"
    minneapple_dir = prediction_dir / task_name
    minneapple_dir.mkdir(parents=True)

    # Convert the predictions from TAO to Lightly
    for tao_prediction_file in tqdm(input_dir.glob("*.txt")):

        contents = np.genfromtxt(tao_prediction_file)
        if len(contents.shape) > 1:

            lightly_prediction = {
                "file_name": str(tao_prediction_file.with_suffix(".png")),  # TODO
                "predictions": [],
            }

            for (
                x0,
                y0,
                x1,
                y1,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                score,
            ) in contents[:, 4:]:
                lightly_prediction["predictions"].append(
                    {
                        "category_id": 0,  # everything is an apple
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": score,
                    }
                )

            lightly_prediction_file = (
                minneapple_dir / tao_prediction_file.with_suffix(".json").name
            )
            with lightly_prediction_file.open("w") as f:
                json.dump(lightly_prediction, f)

    # Add tasks and schema
    tasks = [task_name]
    task_file = prediction_dir / "tasks.json"
    with task_file.open("w") as f:
        json.dump(tasks, f)

    schema = {
        "task_type": "object-detection",
        "categories": [
            {
                "id": 0,
                "name": "Apple",
            }
        ],
    }
    schema_file = minneapple_dir / "schema.json"
    with schema_file.open("w") as f:
        json.dump(schema, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    args = parser.parse_args()

    tao_to_lightly(Path(args.input_dir))

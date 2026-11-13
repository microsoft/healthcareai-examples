import argparse
from pathlib import Path

import mlflow.pyfunc
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file-column", action="append", default=[])
    args = parser.parse_args()

    inputs = pd.read_csv(args.data / "data.csv")
    row_ids = inputs.pop("_row_id")
    for column in args.file_column:
        inputs[column] = inputs[column].map(
            lambda relative_path: (args.data / relative_path).read_bytes()
        )
    model = mlflow.pyfunc.load_model(args.model)
    predictions = model.predict(inputs)
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("The MLflow model must return a pandas DataFrame")
    if len(predictions) != len(row_ids):
        raise ValueError(
            "The model returned a different number of rows than it received"
        )

    predictions.insert(0, "_row_id", row_ids.to_numpy())
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)


if __name__ == "__main__":
    main()

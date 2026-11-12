import pytest
import torch

from model_library.utils.inference import perform_inference


class LogitModel(torch.nn.Module):
    def forward(self, features):
        return features, features


@pytest.mark.parametrize("batch_size", [1, 2])
def test_inference_preserves_per_sample_labels(batch_size):
    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 3.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    labels = torch.tensor([2, 0, 1, 0, 2])
    names = ["first", "second", "third", "fourth", "fifth"]
    loader = torch.utils.data.DataLoader(
        list(zip(logits, labels, names)), batch_size=batch_size, shuffle=False
    )

    predictions = perform_inference(LogitModel(), loader)

    assert [row["Name"] for row in predictions] == names
    assert [row["Label"] for row in predictions] == labels.tolist()
    assert [row["PredictedClass"] for row in predictions] == [0, 2, 1, 0, 1]
    assert all(isinstance(row["Label"], int) for row in predictions)
    assert [row["Probability"] for row in predictions] == pytest.approx(
        torch.softmax(logits, dim=1).max(dim=1).values.tolist()
    )

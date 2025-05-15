import torch
from PIL import Image as PILImage
import numpy as np
import pandas as pd
from datasets import Dataset, Features, Image, ClassLabel


def upload_tensor_dataset_to_hub(
    x_tensor: torch.Tensor,
    dataset_repo: str,
    token: str,
    config_name: str,
    label_source="hf://datasets/cat-claws/poison/cifar10-4-huang2021unlearnable/train-00000-of-00001.parquet",
    class_names=None,
    private=False
):
    """
    Converts a torch tensor (N, 3, 32, 32) and a label source into a HuggingFace dataset, then uploads it.

    Args:
        x_tensor (torch.Tensor): Tensor of shape (N, 3, H, W) in [0, 1] range.
        label_source (str): Path to Parquet file or list of labels.
        dataset_repo (str): Hugging Face repo name, e.g., "your-username/my-dataset".
        class_names (List[str], optional): List of class label names. Defaults to CIFAR-10 labels.
        private (bool): Whether to upload as a private dataset.
    """

    if class_names is None:
        class_names = [
            "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
        ]

    # Convert tensor to uint8 images (N, H, W, C)
    x_uint8 = x_tensor.mul(255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    images = [PILImage.fromarray(img) for img in x_uint8]

    # Load labels
    if isinstance(label_source, str):
        labels = pd.read_parquet(label_source)['label'].tolist()
    elif isinstance(label_source, list):
        labels = label_source
    else:
        raise ValueError("label_source must be a path to parquet file or a list of labels.")

    assert len(images) == len(labels), "Mismatch between number of images and labels."

    # Build Hugging Face dataset
    features = Features({
        "image": Image(),
        "label": ClassLabel(names=class_names)
    })

    dataset = Dataset.from_dict({
        "image": images,
        "label": labels
    }, features=features)

    # Push to hub
    dataset.push_to_hub(dataset_repo, config_name= config_name, private=private, token = token)

import requests
import torch
import torchvision.models as models
from robustbench.utils import load_model


def get_labels_map() -> dict:
    """Download and return the ImageNet label mapping."""
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    response = requests.get(url)
    response.raise_for_status()
    return {idx: label for idx, label in enumerate(response.json().values())}


def load_standard_model() -> torch.nn.Module:
    """Load and return the pretrained standard ResNet-50 model."""
    model = models.resnet50(pretrained=True)
    model.eval()
    return model


def load_robust_model() -> torch.nn.Module:
    """Load and return a pretrained robust ResNet-50 model."""
    model = load_model("Salman2020Do_50_2", dataset="imagenet", threat_model="Linf")
    model.eval()
    return model


def find_class_index(labels_map: dict, keyword: str) -> int:
    """
    Find the index of a class in the labels map by keyword (case-insensitive).
    
    Parameters:
        labels_map (dict): ImageNet labels dictionary.
        keyword (str): Partial name to search for.
    
    Returns:
        int: Class index matching the keyword.

    Raises:
        ValueError: If no class contains the keyword.
    """
    for idx, (_, class_name) in labels_map.items():
        if keyword.lower() in class_name.lower():
            print(f"Found class: {class_name} (index: {idx})")
            return idx
    raise ValueError(f"No class found containing keyword: {keyword}")

import os
import random
import shutil
from typing import List, Tuple, Dict

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchattacks import PGD
from tqdm import tqdm


def preprocess_images(image_paths: List[str]) -> torch.Tensor:
    """Preprocess images for the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    images = [transform(Image.open(path).convert("RGB")) for path in image_paths]
    return torch.stack(images)


def get_predictions(model: torch.nn.Module, images: torch.Tensor, labels_map: Dict[int, Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Get top-1 class predictions from the model."""
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    return [labels_map[idx.item()] for idx in predicted]


def apply_pgd_attack(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor,
                     epsilon: float = 0.03, alpha: float = 0.01, iters: int = 40) -> torch.Tensor:
    """Apply PGD attack to a batch of images."""
    attack = PGD(model, eps=epsilon, alpha=alpha, steps=iters)
    return attack(images, labels)


def visualize_results(originals: torch.Tensor, perturbations: torch.Tensor, adversarials: torch.Tensor,
                      model_name: str, labels_original: List[Tuple[str, str]], labels_adversarial: List[Tuple[str, str]]) -> None:
    """Visualize the original, perturbation, and adversarial images."""
    batch_size = originals.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    fig.suptitle(f"Model: {model_name}", fontsize=16)

    perturbations_normalized = (perturbations - perturbations.min()) / (perturbations.max() - perturbations.min() + 1e-8)

    for i in range(batch_size):
        titles = [
            f"Original: {labels_original[i][1]}",
            "Perturbation",
            f"Adversarial: {labels_adversarial[i][1]}"
        ]
        for j, img, title in zip(range(3), [originals[i], perturbations_normalized[i], adversarials[i]], titles):
            ax = axes[i, j] if batch_size > 1 else axes[j]
            ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
            ax.set_title(title)
            ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def get_top5_predictions(model: torch.nn.Module, images: torch.Tensor,
                         labels_map: Dict[int, Tuple[str, str]]) -> List[Tuple[str, float]]:
    """Get top-5 predictions with probabilities."""
    outputs = model(images)
    probs = F.softmax(outputs, dim=1)
    top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
    results = []
    for i in range(5):
        idx = top5_indices[0, i].item()
        prob = top5_probs[0, i].item()
        results.append((labels_map[idx], prob))
    return results


def print_top5_predictions(model: torch.nn.Module, image: torch.Tensor, labels_map: Dict[int, Tuple[str, str]]) -> None:
    """Print top-5 predictions before and after PGD attack."""
    print("\nOriginal Image Top-5 Predictions:")
    top5_original = get_top5_predictions(model, image, labels_map)
    for label, prob in top5_original:
        print(f"{label[1]}: {prob:.4f}")

    top1_label = top5_original[0][0]
    label_idx = next(idx for idx, lbl in labels_map.items() if lbl == top1_label)
    label_tensor = torch.tensor([label_idx])

    perturbed = apply_pgd_attack(model, image, label_tensor)

    print("\nPerturbed Image Top-5 Predictions:")
    top5_perturbed = get_top5_predictions(model, perturbed, labels_map)
    for label, prob in top5_perturbed:
        print(f"{label[1]}: {prob:.4f}")


def create_targeted_adversarials(model: torch.nn.Module, class_id_source: str, class_id_output: str,
                                 num_images: int, labels_map: Dict[int, Tuple[str, str]],
                                 output_dir: str = "./adv_results", tinyimagenet_dir: str = './tiny-imagenet-200') -> None:
    """
    Create targeted adversarial examples between two ImageNet classes.

    Saves: original images, adversarial images, and unrelated 'other' images.
    """
    adv_dir = os.path.join(output_dir, "adv_ex")
    original_dir = os.path.join(output_dir, "original")
    others_dir = os.path.join(output_dir, "others")
    os.makedirs(adv_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(others_dir, exist_ok=True)

    source_dir = os.path.join(tinyimagenet_dir, 'train', class_id_source, 'images')
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Class {class_id_source} not found in dataset")

    all_images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

    if len(all_images) < 2 * num_images:
        raise ValueError(f"Not enough images in {class_id_source}: found {len(all_images)}, need {2 * num_images}")

    selected = random.sample(all_images, 2 * num_images)
    original_images = selected[:num_images]
    other_images = selected[num_images:]

    # Get target class index from output WordNet ID
    target_idx = next((idx for idx, label in labels_map.items() if class_id_output in label[0]), None)
    if target_idx is None:
        raise ValueError(f"Target class ID {class_id_output} not found in labels_map")

    attack = PGD(model, eps=8/255, alpha=2/255, steps=40)
    attack.set_mode_targeted_by_label()

    for img_file in tqdm(original_images, desc="Creating adversarial examples", unit="image"):
        img_path = os.path.join(source_dir, img_file)
        image = preprocess_images([img_path])
        adv_image = attack(image, torch.tensor([target_idx]))
        torchvision.utils.save_image(adv_image, os.path.join(adv_dir, img_file))
        shutil.copy(img_path, os.path.join(original_dir, img_file))

    for img_file in other_images:
        shutil.copy(os.path.join(source_dir, img_file), os.path.join(others_dir, img_file))

    print(f"Created {num_images} adversarial examples in {output_dir}")

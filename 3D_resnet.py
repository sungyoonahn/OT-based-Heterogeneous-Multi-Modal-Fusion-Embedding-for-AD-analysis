"""Train a vanilla 3D ResNet on T1 MRI NIfTI volumes.

Expected directory layout (default ``/home/prml/RIMA/datasets/ADNI``):

```
ADNI/
  AD_MRI_130_FIN/
    ADNI/
      <patient_id>/
        <scan_type>/
          <date>/
            <image_id>/
              *.nii or *.nii.gz
  CN_MRI_229_FIN/
    ADNI/
      <patient_id>/
        <scan_type>/
          <date>/
            <image_id>/
              *.nii or *.nii.gz
  MCI_MRI_86_FIN/
    ADNI/
      <patient_id>/
        <scan_type>/
          <date>/
            <image_id>/
              *.nii or *.nii.gz
```

The script recursively searches for .nii files in nested subdirectories,
builds a simple train/validation split, normalizes each volume
individually (zero mean, unit variance), resizes to a fixed 3D shape, and
trains a 3D ResNet-18 with a single input channel.
"""

import argparse
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.models.video.resnet import BasicBlock, R2Plus1dStem, Conv2Plus1D
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


# CLASS_NAMES will be auto-detected based on available directories
# Supports both MRI and PET datasets
CLASS_NAMES_MRI = {
    "AD_MRI_130_FIN": 0,
    "CN_MRI_229_FIN": 1,
    "MCI_MRI_86_FIN": 2,
}

CLASS_NAMES_PET = {
    "AD_PET_130_FIN": 0,
    "CN_PET_229_FIN": 1,
    "MCI_PET_86_FIN": 2,
}


def detect_class_names(root_dir: str) -> Dict[str, int]:
    """Auto-detect whether we're using MRI or PET based on available directories."""
    # Check which directories exist
    mri_exists = any(os.path.isdir(os.path.join(root_dir, d)) for d in CLASS_NAMES_MRI.keys())
    pet_exists = any(os.path.isdir(os.path.join(root_dir, d)) for d in CLASS_NAMES_PET.keys())
    
    if mri_exists and not pet_exists:
        print("Detected MRI dataset")
        return CLASS_NAMES_MRI
    elif pet_exists and not mri_exists:
        print("Detected PET dataset")
        return CLASS_NAMES_PET
    elif mri_exists and pet_exists:
        # Both exist, check which has more files
        mri_count = sum(1 for d in CLASS_NAMES_MRI.keys() 
                       if os.path.isdir(os.path.join(root_dir, d)))
        pet_count = sum(1 for d in CLASS_NAMES_PET.keys() 
                       if os.path.isdir(os.path.join(root_dir, d)))
        if mri_count >= pet_count:
            print("Detected MRI dataset (both MRI and PET directories found, using MRI)")
            return CLASS_NAMES_MRI
        else:
            print("Detected PET dataset (both MRI and PET directories found, using PET)")
            return CLASS_NAMES_PET
    else:
        raise RuntimeError(f"No MRI or PET directories found in {root_dir}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NiftiDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        target_shape: Tuple[int, int, int],
        class_names: Dict[str, int],
        augment: bool = False,
        max_samples_per_class: int | None = None,
        patient_ids_filter: Dict[str, List[str]] | None = None,
        seed: int = 42,
    ) -> None:
        self.root_dir = root_dir
        self.target_shape = target_shape
        self.class_names = class_names
        self.augment = augment
        self.max_samples_per_class = max_samples_per_class
        self.patient_ids_filter = patient_ids_filter
        self.seed = seed
        self.samples: List[Tuple[str, int]] = []
        self.patient_ids_used: Dict[str, List[str]] = {class_name: [] for class_name in self.class_names.keys()}
        self._collect_samples()

    def _extract_patient_id(self, file_path: str) -> str | None:
        """Extract patient ID from file path (e.g., '137_S_4672' from path)."""
        parts = file_path.split(os.sep)
        # Look for pattern like XXX_S_XXXX in the path
        for part in parts:
            if '_S_' in part:
                return part
        return None

    def _collect_samples(self) -> None:
        # First, collect all samples grouped by class
        samples_by_class: Dict[str, List[Tuple[str, int, str]]] = {
            class_name: [] for class_name in self.class_names.keys()
        }
        
        for class_dir, label in self.class_names.items():
            dir_path = os.path.join(self.root_dir, class_dir)
            if not os.path.isdir(dir_path):
                continue
            # Recursively search for .nii files in all subdirectories
            for root, dirs, files in os.walk(dir_path):
                for file_name in files:
                    if file_name.endswith((".nii", ".nii.gz")):
                        file_path = os.path.join(root, file_name)
                        patient_id = self._extract_patient_id(file_path)
                        if patient_id:
                            samples_by_class[class_dir].append((file_path, label, patient_id))
        
        # Filter and limit samples per class
        rng = random.Random(self.seed)
        for class_dir, class_samples in samples_by_class.items():
            if self.patient_ids_filter and class_dir in self.patient_ids_filter:
                # Filter by specific patient IDs
                filtered_samples = [
                    (path, label, pid) for path, label, pid in class_samples
                    if pid in self.patient_ids_filter[class_dir]
                ]
                class_samples = filtered_samples
            elif self.max_samples_per_class:
                # Group by patient ID to ensure we select diverse patients
                patient_groups: Dict[str, List[Tuple[str, int, str]]] = {}
                for path, label, pid in class_samples:
                    if pid not in patient_groups:
                        patient_groups[pid] = []
                    patient_groups[pid].append((path, label, pid))
                
                # Randomly select patients until we have enough samples
                patient_ids = list(patient_groups.keys())
                rng.shuffle(patient_ids)
                
                selected_samples = []
                for pid in patient_ids:
                    if len(selected_samples) >= self.max_samples_per_class:
                        break
                    # Take one sample per patient
                    selected_samples.extend(patient_groups[pid][:1])
                
                class_samples = selected_samples[:self.max_samples_per_class]
            
            # Add to final samples list and track patient IDs
            for path, label, pid in class_samples:
                self.samples.append((path, label))
                if pid not in self.patient_ids_used[class_dir]:
                    self.patient_ids_used[class_dir].append(pid)
        
        if not self.samples:
            raise RuntimeError(f"No NIfTI files found under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_volume(self, volume: torch.Tensor) -> torch.Tensor:
        # volume: (1, D, H, W)
        volume = volume.unsqueeze(0)  # (1, 1, D, H, W)
        volume = F.interpolate(
            volume,
            size=self.target_shape,
            mode="trilinear",
            align_corners=False,
        )
        return volume.squeeze(0)

    def _augment(self, volume: torch.Tensor) -> torch.Tensor:
        # Random flips on each spatial dimension.
        if random.random() < 0.5:
            volume = torch.flip(volume, dims=[1])
        if random.random() < 0.5:
            volume = torch.flip(volume, dims=[2])
        if random.random() < 0.5:
            volume = torch.flip(volume, dims=[3])
        return volume

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = nib.load(path).get_fdata().astype(np.float32)
        img = np.nan_to_num(img)
        volume = torch.from_numpy(img).unsqueeze(0)  # (1, D, H, W)
        volume = self._resize_volume(volume)

        # Per-volume z-score normalization.
        mean = volume.mean()
        std = volume.std()
        volume = (volume - mean) / (std + 1e-5)

        if self.augment:
            volume = self._augment(volume)

        return volume, label


class Bottleneck3D(nn.Module):
    """3D Bottleneck block for ResNet-50."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    """3D ResNet-50 architecture."""

    def __init__(self, block, layers, num_classes=3):
        super().__init__()
        self.inplanes = 64
        
        # Stem: initial conv layer
        self.stem = nn.Sequential(
            nn.Conv3d(
                1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def build_model(num_classes: int, model_depth: int = 50) -> nn.Module:
    """Build a 3D ResNet model."""
    if model_depth == 10:
        layers = [1, 1, 1, 1]
        block = BasicBlock
    elif model_depth == 18:
        layers = [2, 2, 2, 2]
        block = BasicBlock
    elif model_depth == 34:
        layers = [3, 4, 6, 3]
        block = BasicBlock
    elif model_depth == 50:
        layers = [3, 4, 6, 3]
        block = Bottleneck3D
    elif model_depth == 101:
        layers = [3, 4, 23, 3]
        block = Bottleneck3D
    elif model_depth == 152:
        layers = [3, 8, 36, 3]
        block = Bottleneck3D
    elif model_depth == 200:
        layers = [3, 24, 36, 3]
        block = Bottleneck3D
    else:
        raise ValueError(f"Unsupported model depth: {model_depth}")

    model = ResNet3D(block, layers, num_classes=num_classes)
    return model


def split_dataset(
    dataset: Dataset,
    val_fraction: float,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    """Perform stratified split to ensure balanced classes in train/val sets."""
    # Group indices by label
    indices_by_label: Dict[int, List[int]] = {}
    # Access the underlying samples list from NiftiDataset
    # dataset.samples is a list of (path, label)
    for idx, (_, label) in enumerate(dataset.samples):
        if label not in indices_by_label:
            indices_by_label[label] = []
        indices_by_label[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    rng = random.Random(seed)
    
    for label, indices in indices_by_label.items():
        # Shuffle indices for this class
        rng.shuffle(indices)
        
        # Calculate split sizes for this class
        n_val = int(len(indices) * val_fraction)
        
        # Append to respective lists
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
        
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)
        
        # Update progress bar with current metrics
        current_loss = total_loss / total_samples
        current_acc = total_correct / total_samples
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)
            
            # Collect predictions and targets for metrics
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            
            # Update progress bar with current metrics
            current_loss = total_loss / total_samples
            current_acc = total_correct / total_samples
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, all_preds, all_targets


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a vanilla 3D ResNet on MRI NIfTI volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/prml/RIMA/datasets/ADNI",
        help="Root directory containing AD_MRI_130_FIN, CN_MRI_229_FIN, and MCI_MRI_86_FIN folders.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of data used for validation.",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        metavar=("D", "H", "W"),
        help="Depth, height, and width to resize each volume.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--save-path",
        type=str,
        default="results/ADNI_MRI_3D_RESNET",
        help="Directory to save results and model checkpoint.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable simple random flips during training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device to use.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Maximum number of samples to use per class. If not specified, use all samples.",
    )
    parser.add_argument(
        "--load-patient-ids",
        type=str,
        default=None,
        help="Path to JSON file containing patient IDs to use (for matching MRI/PET cohorts).",
    )
    parser.add_argument(
        "--model-depth",
        type=int,
        default=101,
        choices=[10, 18, 34, 50, 101, 152, 200],
        help="Depth of the ResNet model (default: 101).",
    )
    return parser.parse_args(argv)


def calculate_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    """Calculate precision, recall, F1, and specificity for multi-class classification."""
    # Calculate precision, recall, F1 (macro-averaged)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Calculate specificity for each class and average
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    
    avg_specificity = np.mean(specificities)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': avg_specificity
    }


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Dict[str, int],
    save_path: str,
) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    # Get class names in order of their indices
    labels = sorted(class_names.keys(), key=lambda k: class_names[k])
    # Shorten labels for plotting if needed (remove _MRI_... / _PET_...)
    short_labels = [l.split('_')[0] for l in labels]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=short_labels,
        yticklabels=short_labels,
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    results_file = os.path.join(args.save_path, "results.txt")
    model_path = os.path.join(args.save_path, "best_model.pth")

    device = torch.device(args.device)

    # Auto-detect MRI or PET dataset
    class_names = detect_class_names(args.data_dir)

    # Load patient IDs if specified
    patient_ids_filter = None
    if args.load_patient_ids:
        print(f"Loading patient IDs from {args.load_patient_ids}")
        with open(args.load_patient_ids, 'r') as f:
            patient_ids_filter = json.load(f)
        print(f"Loaded patient IDs for {len(patient_ids_filter)} classes")

    full_dataset = NiftiDataset(
        root_dir=args.data_dir,
        target_shape=tuple(args.target_shape),
        class_names=class_names,
        augment=args.augment,
        max_samples_per_class=args.max_samples_per_class,
        patient_ids_filter=patient_ids_filter,
        seed=args.seed,
    )
    
    # Save patient IDs used
    patient_ids_file = os.path.join(args.save_path, "patient_ids.json")
    with open(patient_ids_file, 'w') as f:
        json.dump(full_dataset.patient_ids_used, f, indent=2)
    print(f"Saved patient IDs to {patient_ids_file}")
    
    # Print sample counts per class
    print("\nSamples per class:")
    for class_name in class_names.keys():
        count = len(full_dataset.patient_ids_used[class_name])
        print(f"  {class_name}: {count} patients")
    print(f"Total samples: {len(full_dataset)}\n")
    train_dataset, val_dataset = split_dataset(full_dataset, args.val_fraction, args.seed)

    # Verify split balance
    print("\nVerifying split balance:")
    def count_labels(subset, name):
        counts = {}
        for idx in subset.indices:
            _, label = subset.dataset.samples[idx]
            class_name = [k for k, v in class_names.items() if v == label][0]
            counts[class_name] = counts.get(class_name, 0) + 1
        print(f"  {name} set:")
        for class_name in sorted(counts.keys()):
            print(f"    {class_name}: {counts[class_name]}")
            
    count_labels(train_dataset, "Train")
    count_labels(val_dataset, "Validation")
    print("")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(num_classes=len(class_names), model_depth=args.model_depth).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')  # Track best validation loss (lower is better)
    
    # Open results file
    with open(results_file, "w") as f:
        f.write("3D ResNet Training Results - ADNI MRI Dataset\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: {args.data_dir}\n")
        f.write(f"Train/Val Split: {1-args.val_fraction:.1%}/{args.val_fraction:.1%}\n")
        f.write(f"Total Samples: {len(full_dataset)}\n")
        f.write(f"Train Samples: {len(train_dataset)}\n")
        f.write(f"Val Samples: {len(val_dataset)}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Target Shape: {args.target_shape}\n")
        f.write(f"Device: {device}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<11} {'Val Loss':<12} {'Val Acc':<11} "
                f"{'Precision':<11} {'Recall':<11} {'F1 Score':<11} {'Specificity':<12}\n")
        f.write("-" * 120 + "\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        
        # Calculate additional metrics
        metrics = calculate_metrics(val_targets, val_preds, len(class_names))
        
        # Print to console
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} spec={metrics['specificity']:.4f}"
        )
        
        # Write to results file
        with open(results_file, "a") as f:
            f.write(f"{epoch:<6} {train_loss:<12.4f} {train_acc:<11.4f} {val_loss:<12.4f} {val_acc:<11.4f} "
                    f"{metrics['precision']:<11.4f} {metrics['recall']:<11.4f} "
                    f"{metrics['f1']:<11.4f} {metrics['specificity']:<12.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "metrics": metrics,
                    "args": vars(args),
                },
                model_path,
            )
            print(f"Saved new best model to {model_path} (val_loss={val_loss:.4f})")
    
    # Write final summary
    with open(results_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"Best model saved to: {model_path}\n")

    # Load best model and generate confusion matrix
    print(f"Loading best model from {model_path} for confusion matrix...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    _, _, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
    cm_path = os.path.join(args.save_path, "confusion_matrix.png")
    save_confusion_matrix(val_targets, val_preds, class_names, cm_path)
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()

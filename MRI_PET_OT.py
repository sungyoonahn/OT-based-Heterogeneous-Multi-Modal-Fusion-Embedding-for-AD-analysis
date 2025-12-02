"""Multimodal MRI-PET Classification using Optimal Transport

This script trains a multimodal model that combines MRI and PET scans using
optimal transport for feature alignment. It uses 3D ResNet-50 backbones for
both modalities and fuses them using optimal transport coupling.

Expected directory layout:
ADNI/
  AD_MRI_130_FIN/  (MRI scans)
  CN_MRI_229_FIN/
  MCI_MRI_86_FIN/
  AD_PET_130_FIN/  (PET scans)
  CN_PET_229_FIN/
  MCI_PET_86_FIN/
"""

import os
# JAX/XLA configuration for CPU usage
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_DISABLE_JIT", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("JAX_ENABLE_X64", "True")  # Enable x64 for stability
existing_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
threading_flags = "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
if "--xla_force_host_platform_device_count" not in existing_xla_flags:
    os.environ["XLA_FLAGS"] = " ".join(filter(None, [existing_xla_flags, threading_flags]))

import argparse
import json
import random
from typing import Dict, List, Tuple, Any
from numbers import Number
import time

import matplotlib.pyplot as plt
import seaborn as sns

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision.models.video.resnet import BasicBlock
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from perturbot.match import get_coupling_fot

# OTT imports for custom solver
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import gromov_wasserstein
from ott.solvers.linear import sinkhorn

# Class names for MRI and PET
CLASS_NAMES_MRI = {
    "AD_MRI_130_FIN": 0,
    "CN_MRI_229_FIN": 1,
}

CLASS_NAMES_PET = {
    "AD_PET_130_FIN": 0,
    "CN_PET_229_FIN": 1,
}

def get_coupling_egw_ott_fixed(
    data: Tuple[Dict[Number, np.array], Dict[Number, np.array]], eps: float = 5e-3
) -> Tuple[Dict[Number, np.array], Dict]:
    """Fixed version of get_coupling_egw_ott that works with newer OTT versions."""
    X_dict = data[0]
    Y_dict = data[1]
    labels = X_dict.keys()
    Ts = {}
    log = {}
    for l in labels:
        log[l] = {}
        start = time.time()

        # Ensure inputs are JAX arrays
        x_data = jnp.array(X_dict[l])
        y_data = jnp.array(Y_dict[l])
        
        # Check for NaNs
        if jnp.isnan(x_data).any() or jnp.isnan(y_data).any():
            print(f"Warning: NaNs detected in features for label {l}")
            x_data = jnp.nan_to_num(x_data)
            y_data = jnp.nan_to_num(y_data)

        geom_xx = pointcloud.PointCloud(x=x_data, y=x_data, scale_cost="max_cost")
        geom_yy = pointcloud.PointCloud(x=y_data, y=y_data, scale_cost="max_cost")

        cost_time = time.time() - start
        start = time.time()

        prob = quadratic_problem.QuadraticProblem(
            geom_xx,
            geom_yy,
        )

        # Instantiate Gromov-Wasserstein solver with linear_solver
        solver = gromov_wasserstein.GromovWasserstein(
            epsilon=eps, 
            store_inner_errors=True, 
            max_iterations=1000,
            linear_solver=sinkhorn.Sinkhorn()
        )

        out = solver(prob)

        end = time.time()
        Ts[l] = np.array(out.matrix)

        has_converged = bool(out.linear_convergence[out.n_iters - 1])
        log[l]["n_iters_outer"] = out.n_iters
        log[l]["converged_inner"] = has_converged
        log[l]["converged_outer"] = out.converged
        log[l]["GW cost"] = out.reg_gw_cost
        log[l]["time"] = end - start
        log[l]["cost_time"] = cost_time
    return Ts, log


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


def split_dataset(
    dataset: Dataset,
    val_fraction: float,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    """Perform stratified split to ensure balanced classes in train/val sets."""
    # Group indices by label
    indices_by_label: Dict[int, List[int]] = {}
    # Access the underlying samples list from MultimodalNiftiDataset
    # dataset.samples is a list of (mri_path, pet_path, label)
    for idx, (_, _, label) in enumerate(dataset.samples):
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultimodalNiftiDataset(Dataset):
    """Dataset that loads paired MRI and PET scans."""
    
    def __init__(
        self,
        root_dir: str,
        target_shape: Tuple[int, int, int],
        augment: bool = False,
        max_samples_per_class: int | None = None,
        patient_ids_filter: Dict[str, List[str]] | None = None,
        seed: int = 42,
    ) -> None:
        self.root_dir = root_dir
        self.target_shape = target_shape
        self.augment = augment
        self.max_samples_per_class = max_samples_per_class
        self.patient_ids_filter = patient_ids_filter
        self.seed = seed
        self.samples: List[Tuple[str, str, int]] = []  # (mri_path, pet_path, label)
        self.patient_ids_used: Dict[str, List[str]] = {class_name: [] for class_name in CLASS_NAMES_MRI.keys()}
        self._collect_samples()

    def _collect_samples(self) -> None:
        """Collect paired MRI and PET samples."""
        # Collect MRI files
        mri_files = {}
        for class_dir, label in CLASS_NAMES_MRI.items():
            dir_path = os.path.join(self.root_dir, class_dir)
            if not os.path.isdir(dir_path):
                continue
            for root, dirs, files in os.walk(dir_path):
                for file_name in files:
                    if file_name.endswith((".nii", ".nii.gz")):
                        # Extract patient ID from full path
                        full_path = os.path.join(root, file_name)
                        patient_id = self._extract_patient_id(full_path)
                        if patient_id:
                            mri_files[patient_id] = (full_path, label)
        
        # Collect PET files and match with MRI
        # We collect by class, assuming MRI and PET have same class structure
        
        # Temporary storage for samples by class to apply filtering
        samples_by_class: Dict[str, List[Tuple[str, str, int, str]]] = {
            class_name: [] for class_name in CLASS_NAMES_MRI.keys()
        }

        for class_dir_pet, label in CLASS_NAMES_PET.items():
            # Find corresponding MRI class dir
            # Assuming class names match pattern like AD_PET_... -> AD_MRI_...
            # But we can just map by label since we have CLASS_NAMES_MRI
            class_dir_mri = [k for k, v in CLASS_NAMES_MRI.items() if v == label][0]
            
            dir_path = os.path.join(self.root_dir, class_dir_pet)
            if not os.path.isdir(dir_path):
                continue
                
            for root, dirs, files in os.walk(dir_path):
                for file_name in files:
                    if file_name.endswith((".nii", ".nii.gz")):
                        full_path_pet = os.path.join(root, file_name)
                        patient_id = self._extract_patient_id(full_path_pet)
                        
                        if patient_id and patient_id in mri_files:
                            mri_path, mri_label = mri_files[patient_id]
                            # Ensure labels match
                            if mri_label == label:
                                samples_by_class[class_dir_mri].append((mri_path, full_path_pet, label, patient_id))

        # Filter and limit samples per class
        rng = random.Random(self.seed)
        for class_dir, class_samples in samples_by_class.items():
            if self.patient_ids_filter and class_dir in self.patient_ids_filter:
                # Filter by specific patient IDs
                filtered_samples = [
                    (mri, pet, lbl, pid) for mri, pet, lbl, pid in class_samples
                    if pid in self.patient_ids_filter[class_dir]
                ]
                class_samples = filtered_samples
            elif self.max_samples_per_class:
                # Group by patient ID to ensure we select diverse patients
                # (Though here we likely have one pair per patient anyway)
                patient_groups = {}
                for mri, pet, lbl, pid in class_samples:
                    if pid not in patient_groups:
                        patient_groups[pid] = []
                    patient_groups[pid].append((mri, pet, lbl, pid))
                
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
            for mri, pet, lbl, pid in class_samples:
                self.samples.append((mri, pet, lbl))
                if pid not in self.patient_ids_used[class_dir]:
                    self.patient_ids_used[class_dir].append(pid)
        
        if not self.samples:
            raise RuntimeError(f"No paired MRI-PET files found under {self.root_dir}")
        
        print(f"Found {len(self.samples)} paired MRI-PET samples")

    def _extract_patient_id(self, path: str) -> str:
        """Extract patient ID from file path or filename."""
        import re
        # First try to extract from directory names
        parts = path.split(os.sep)
        for part in parts:
            # Match pattern: XXX_S_XXXX (e.g., 002_S_5018, 137_S_4672)
            if re.match(r'^\d{3}_S_\d{4}$', part):
                return part
        
        # If not found in directories, try to extract from filename
        # PET files are named like: XXX_S_XXXX_AV45.nii
        filename = os.path.basename(path)
        match = re.match(r'^(\d{3}_S_\d{4})_', filename)
        if match:
            return match.group(1)
        
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def _resize_volume(self, volume: torch.Tensor) -> torch.Tensor:
        volume = volume.unsqueeze(0)  # (1, 1, D, H, W)
        volume = F.interpolate(
            volume,
            size=self.target_shape,
            mode="trilinear",
            align_corners=False,
        )
        return volume.squeeze(0)

    def _augment(self, volume: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            volume = torch.flip(volume, dims=[1])
        if random.random() < 0.5:
            volume = torch.flip(volume, dims=[2])
        if random.random() < 0.5:
            volume = torch.flip(volume, dims=[3])
        return volume

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], int]:
        mri_path, pet_path, label = self.samples[index]
        
        # Load MRI
        mri_img = nib.load(mri_path).get_fdata().astype(np.float32)
        mri_img = np.nan_to_num(mri_img)
        mri_volume = torch.from_numpy(mri_img).unsqueeze(0)
        mri_volume = self._resize_volume(mri_volume)
        
        # Load PET
        pet_img = nib.load(pet_path).get_fdata().astype(np.float32)
        pet_img = np.nan_to_num(pet_img)
        pet_volume = torch.from_numpy(pet_img).unsqueeze(0)
        pet_volume = self._resize_volume(pet_volume)
        
        # Normalize each volume
        for vol in [mri_volume, pet_volume]:
            mean = vol.mean()
            std = vol.std()
            vol.sub_(mean).div_(std + 1e-5)
        
        if self.augment:
            mri_volume = self._augment(mri_volume)
            pet_volume = self._augment(pet_volume)
        
        return [mri_volume, pet_volume], label


# 3D ResNet-50 Components
class Bottleneck3D(nn.Module):
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


class ResNet3D_Backbone(nn.Module):
    """3D ResNet-50 backbone for feature extraction."""

    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        
        self.stem = nn.Sequential(
            nn.Conv3d(
                1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

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
        return x


class SelfAttentionBlock(nn.Module):
    """Transformer encoder block for feature fusion."""

    def __init__(self, embed_dim=2048, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        out = self.norm2(x)
        return out


def cosine_loss(x, y):
    """Cosine similarity loss for feature alignment."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return 1 - F.cosine_similarity(x, y).mean()


class SelfAttentionBlock(nn.Module):
    """Transformer encoder block for feature fusion."""
    def __init__(self, embed_dim=2048, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        return self.norm2(x)


class MultimodalMRI_PET_OT(nn.Module):
    """Multimodal MRI-PET model with Optimal Transport fusion."""

    def __init__(self, num_classes=3, model_depth=50):
        super().__init__()
        self.num_classes = num_classes
        
        # Determine block and layers based on depth
        if model_depth == 10:
            layers = [1, 1, 1, 1]
            block = BasicBlock
            expansion = 1
        elif model_depth == 18:
            layers = [2, 2, 2, 2]
            block = BasicBlock
            expansion = 1
        elif model_depth == 34:
            layers = [3, 4, 6, 3]
            block = BasicBlock
            expansion = 1
        elif model_depth == 50:
            layers = [3, 4, 6, 3]
            block = Bottleneck3D
            expansion = 4
        elif model_depth == 101:
            layers = [3, 4, 23, 3]
            block = Bottleneck3D
            expansion = 4
        elif model_depth == 152:
            layers = [3, 8, 36, 3]
            block = Bottleneck3D
            expansion = 4
        elif model_depth == 200:
            layers = [3, 24, 36, 3]
            block = Bottleneck3D
            expansion = 4
        else:
            raise ValueError(f"Unsupported model depth: {model_depth}")

        # Separate backbones for MRI and PET
        self.mri_backbone = ResNet3D_Backbone(block, layers)
        self.pet_backbone = ResNet3D_Backbone(block, layers)
        
        # Feature dimension from ResNet
        feature_dim = 512 * expansion
        
        # Cross-modality projection layers
        self.mri2pet = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        self.pet2mri = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        # Fusion layers
        self.mri_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim),
        )

        self.pet_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim),
        )

        # Self-attention for MRI features
        self.attention_mri = SelfAttentionBlock(embed_dim=feature_dim, num_heads=8, ff_dim=feature_dim, dropout=0.1)

        # Final classifier
        self.fc = nn.Linear(feature_dim * 2, num_classes)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, X, y, T_feature_pet2mri=None, training=False):
        """
        X: List of [mri_volume, pet_volume]
        y: Target labels
        T_feature_pet2mri: Optional pre-computed OT coupling
        training: Boolean flag
        """
        # 1. Extract features
        mri_feat = self.mri_backbone(X[0])
        pet_feat = self.pet_backbone(X[1])

        # 2. Cross-modality projection
        mri_proj = self.mri2pet(mri_feat)
        pet_proj = self.pet2mri(pet_feat)

        # 3. Fusion
        mri_fused = self.mri_fusion(torch.cat([mri_feat, mri_proj], dim=1))
        pet_fused = self.pet_fusion(torch.cat([pet_feat, pet_proj], dim=1))

        # 4. Self-attention for MRI
        # Reshape for MultiheadAttention: (seq_len, batch, embed_dim) -> (1, B, C)
        attn_input = mri_fused.unsqueeze(0)
        attn_out = self.attention_mri(attn_input)
        attn_out = attn_out.squeeze(0)

        # 5. Classification
        logits = self.fc(torch.cat([attn_out, pet_fused], dim=1))
        
        # 6. Loss calculation
        ce_loss = self.ce_loss(logits, y)
        
        ot_loss = torch.tensor(0.0, device=logits.device)
        
        if training:
            # Compute OT coupling if not provided
            if T_feature_pet2mri is None:
                # Get coupling using FOT (Feature Optimal Transport)
                mri_np = mri_fused.detach().cpu().numpy()
                pet_np = pet_fused.detach().cpu().numpy()
                batch_size = mri_np.shape[0]
                
                # Assume one-to-one correspondence in batch -> Identity coupling for samples
                Ts = np.eye(batch_size) / batch_size
                
                # Wrap in dicts as expected by get_coupling_fot
                data_tuple = ({0: mri_np}, {0: pet_np})
                
                T_feature_pet2mri_np, _ = get_coupling_fot(
                    data_tuple,
                    Ts=Ts,
                    eps=1e-3
                )
                
                # Convert back to tensor and handle NaNs
                T_feature_pet2mri = torch.from_numpy(T_feature_pet2mri_np).float().to(logits.device)
                
                # Numerical stability: replace NaNs with small epsilon
                T_feature_pet2mri = torch.where(
                    torch.isnan(T_feature_pet2mri),
                    torch.full_like(T_feature_pet2mri, 1e-8),
                    T_feature_pet2mri
                )
                
                # Row normalization
                row_sums = T_feature_pet2mri.sum(dim=1, keepdim=True)
                # Avoid division by zero
                row_sums = torch.where(row_sums == 0, torch.full_like(row_sums, 1e-8), row_sums)
                T_feature_pet2mri = T_feature_pet2mri / row_sums

            # Apply OT coupling: map PET features to MRI space
            ot_mri_from_pet = torch.matmul(pet_fused, T_feature_pet2mri.t())
            
            # Calculate OT loss (cosine distance)
            ot_loss = cosine_loss(mri_fused, ot_mri_from_pet)
            
            # Final NaN guard for the loss itself
            if torch.isnan(ot_loss):
                ot_loss = torch.tensor(0.0, device=logits.device)

        return logits, ce_loss, ot_loss

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(y_true: List[int], y_pred: List[int], num_classes: int) -> Dict[str, float]:
    """Calculate comprehensive metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    T_feature_pet2mri=None,
) -> Tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for data, targets in pbar:
        # Move data to device
        for v_num in range(len(data)):
            data[v_num] = data[v_num].float().to(device)
        targets = targets.long().to(device)
        
        current_batch_size = targets.size(0)
        
        optimizer.zero_grad()
        pred, ce_loss, ot_loss = model(X=data, y=targets, T_feature_pet2mri=T_feature_pet2mri, training=True)
        
        loss = ce_loss + ot_loss
        loss.backward()
        optimizer.step()
        
        predicted = pred.argmax(dim=-1)
        total_correct += (predicted == targets).sum().item()
        total_samples += current_batch_size
        loss_meter.update(loss.item(), current_batch_size)
        
        current_acc = total_correct / total_samples
        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{current_acc:.4f}"})

    avg_loss = loss_meter.avg
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    T_feature_pet2mri,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    loss_meter = AverageMeter()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for data, targets in pbar:
            for v_num in range(len(data)):
                data[v_num] = data[v_num].float().to(device)
            targets = targets.long().to(device)

            pred, ce_loss, ot_loss = model(X=data, y=targets, T_feature_pet2mri=T_feature_pet2mri, training=False)
            
            # Total loss for validation monitoring
            loss = ce_loss + ot_loss

            predicted = pred.argmax(dim=-1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            loss_meter.update(loss.item(), targets.size(0))
            
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            
            current_acc = total_correct / total_samples
            pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}", "acc": f"{current_acc:.4f}"})

    avg_loss = loss_meter.avg
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, all_preds, all_targets


def feature_extract(model, data_loader, device):
    """Extract features for OT coupling computation."""
    list_mri_features = []
    list_pet_features = []
    list_labels = []
    
    with torch.no_grad():
        model.eval()
        for data, targets in tqdm(data_loader, desc="Extracting features"):
            for v_num in range(len(data)):
                data[v_num] = data[v_num].float().to(device)
            
            mri_features = model.mri_backbone(data[0])
            pet_features = model.pet_backbone(data[1])
            
            list_mri_features.append(mri_features.cpu().numpy())
            list_pet_features.append(pet_features.cpu().numpy())
            list_labels.append(targets.numpy())
    
    mri_features = np.concatenate(list_mri_features, axis=0)
    mri_features = torch.tensor(mri_features, dtype=torch.float32)
    pet_features = np.concatenate(list_pet_features, axis=0)
    pet_features = torch.tensor(pet_features, dtype=torch.float32)
    labels = np.concatenate(list_labels, axis=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return mri_features, pet_features, labels


def group_features_by_label(y, p, max_samples_per_label=None):
    """Group features by label with optional sampling limit."""
    unique_labels = np.unique(y)
    grouped_features = {int(label): [] for label in unique_labels}
    y_np = y
    p_np = p
    for label, features in zip(y_np, p_np):
        label = int(label)
        grouped_features[label].append(features)
    for label in grouped_features:
        if grouped_features[label]:
            arr = np.stack(grouped_features[label])
            if (
                max_samples_per_label is not None
                and max_samples_per_label > 0
                and arr.shape[0] > max_samples_per_label
            ):
                arr = arr[:max_samples_per_label]
            grouped_features[label] = arr
    return grouped_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multimodal MRI-PET model with Optimal Transport",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/prml/RIMA/datasets/ADNI",
        help="Root directory containing MRI and PET folders",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=(128, 128, 128),
        metavar=("D", "H", "W"),
        help="Target volume shape",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-path",
        type=str,
        default="results/MRI_PET_OT_AD_CN",
        help="Directory to save results",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation",
    )   
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--max-jax-samples",
        type=int,
        default=64,
        help="Max samples per label for OT computation",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=130,
        help="Maximum number of samples to use per class.",
    )
    parser.add_argument(
        "--load-patient-ids",
        type=str,
        default=None,
        help="Path to JSON file containing patient IDs to use.",
    )
    parser.add_argument(
        "--model-depth",
        type=int,
        default=101,
        choices=[10, 18, 34, 50, 101, 152, 200],
        help="Depth of the ResNet model (default: 101).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    results_file = os.path.join(args.save_path, "results.txt")
    model_path = os.path.join(args.save_path, "best_model.pth")

    device = torch.device(args.device)

    # Load patient IDs if specified
    patient_ids_filter = None
    if args.load_patient_ids:
        print(f"Loading patient IDs from {args.load_patient_ids}")
        with open(args.load_patient_ids, 'r') as f:
            patient_ids_filter = json.load(f)
        print(f"Loaded patient IDs for {len(patient_ids_filter)} classes")

    # Load dataset
    full_dataset = MultimodalNiftiDataset(
        root_dir=args.data_dir,
        target_shape=tuple(args.target_shape),
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
    for class_name in CLASS_NAMES_MRI.keys():
        count = len(full_dataset.patient_ids_used[class_name])
        print(f"  {class_name}: {count} patients")
    print(f"Total samples: {len(full_dataset)}\n")

    # Split dataset (Stratified)
    train_dataset, val_dataset = split_dataset(full_dataset, args.val_fraction, args.seed)
    
    # Verify split balance
    print("\nVerifying split balance:")
    def count_labels(subset, name):
        counts = {}
        for idx in subset.indices:
            _, _, label = subset.dataset.samples[idx]
            class_name = [k for k, v in CLASS_NAMES_MRI.items() if v == label][0]
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

    # Build model
    model = MultimodalMRI_PET_OT(num_classes=2, model_depth=args.model_depth).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    
    # Write header
    with open(results_file, "w") as f:
        f.write("Multimodal MRI-PET with Optimal Transport - ADNI Dataset\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: {args.data_dir}\n")
        f.write(f"Train/Val Split: {1-args.val_fraction:.1%}/{args.val_fraction:.1%}\n")
        f.write(f"Total Samples: {len(full_dataset)}\n")
        f.write(f"Train Samples: {len(train_dataset)}\n")
        f.write(f"Val Samples: {len(val_dataset)}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Target Shape: {args.target_shape}\n")
        f.write(f"Model Depth: {args.model_depth}\n")
        f.write(f"Device: {device}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<11} {'Val Loss':<12} {'Val Acc':<11} "
                f"{'Precision':<11} {'Recall':<11} {'F1 Score':<11} {'Specificity':<12}\n")
        f.write("-" * 120 + "\n")

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Extract features for OT coupling
        print("Extracting features for OT coupling...")
        mri_features, pet_features, train_labels = feature_extract(model, train_loader, device)
        
        grouped_mri = group_features_by_label(
            train_labels.cpu().numpy(),
            mri_features.cpu().numpy(),
            max_samples_per_label=args.max_jax_samples,
        )
        grouped_pet = group_features_by_label(
            train_labels.cpu().numpy(),
            pet_features.cpu().numpy(),
            max_samples_per_label=args.max_jax_samples,
        )
        
        # Compute OT coupling for validation
        T_dict_pet2mri, _ = get_coupling_egw_ott_fixed((grouped_pet, grouped_mri))
        T_feature_pet2mri, _ = get_coupling_fot((grouped_pet, grouped_mri), T_dict_pet2mri)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = evaluate(
            model, val_loader, criterion, device, T_feature_pet2mri
        )
        
        # Calculate metrics
        metrics = calculate_metrics(val_targets, val_preds, 2)
        
        # Print results
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} spec={metrics['specificity']:.4f}"
        )
        
        # Save results
        with open(results_file, "a") as f:
            f.write(f"{epoch:<6} {train_loss:<12.4f} {train_acc:<11.4f} {val_loss:<12.4f} {val_acc:<11.4f} "
                    f"{metrics['precision']:<11.4f} {metrics['recall']:<11.4f} "
                    f"{metrics['f1']:<11.4f} {metrics['specificity']:<12.4f}\n")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "metrics": metrics,
                    "args": vars(args),
                },
                model_path,
            )
            print(f"Saved new best model (val_loss={val_loss:.4f})")
        
        scheduler.step(val_loss)
    
    # Final summary
    with open(results_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Best Validation Loss: {best_val_loss:.4f}\n")
        f.write(f"Best model saved to: {model_path}\n")
    
    # Load best model and generate confusion matrix
    print(f"Loading best model from {model_path} for confusion matrix...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Need to re-compute OT coupling for validation with best model features?
    # Ideally yes, but for now we can reuse the last one or recompute.
    # To be precise, we should recompute features with the best model.
    print("Re-extracting features with best model for OT coupling...")
    mri_features, pet_features, train_labels = feature_extract(model, train_loader, device)
    grouped_mri = group_features_by_label(
        train_labels.cpu().numpy(),
        mri_features.cpu().numpy(),
        max_samples_per_label=args.max_jax_samples,
    )
    grouped_pet = group_features_by_label(
        train_labels.cpu().numpy(),
        pet_features.cpu().numpy(),
        max_samples_per_label=args.max_jax_samples,
    )
    T_dict_pet2mri, _ = get_coupling_egw_ott_fixed((grouped_pet, grouped_mri))
    T_feature_pet2mri, _ = get_coupling_fot((grouped_pet, grouped_mri), T_dict_pet2mri)

    _, _, val_preds, val_targets = evaluate(model, val_loader, criterion, device, T_feature_pet2mri)
    cm_path = os.path.join(args.save_path, "confusion_matrix.png")
    save_confusion_matrix(val_targets, val_preds, CLASS_NAMES_MRI, cm_path)
    print(f"Saved confusion matrix to {cm_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

"""
Dataset Loader for Mathematical Symbol Recognition

This module handles loading, preprocessing, and batching of stroke data
for training and evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from collections import Counter
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from stroke_preprocessor import StrokePreprocessor


class StrokeDataset(Dataset):
    """PyTorch Dataset for stroke-based mathematical symbols."""
    
    def __init__(self,
                 data_dir: Path,
                 preprocessor: StrokePreprocessor,
                 label_encoder: Optional[Dict[str, int]] = None,
                 augment: bool = False,
                 cache_dir: Optional[Path] = None,
                 max_samples: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            data_dir (Path): Directory containing InkML files
            preprocessor (StrokePreprocessor): Preprocessor instance
            label_encoder (Optional[Dict]): Label to integer mapping
            augment (bool): Whether to apply data augmentation
            cache_dir (Optional[Path]): Directory for caching processed data
            max_samples (Optional[int]): Maximum number of samples to load
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_samples = max_samples
        
        # Get all InkML files
        self.file_paths = list(self.data_dir.glob("*.inkml"))
        if max_samples:
            self.file_paths = self.file_paths[:max_samples]
        
        print(f"Found {len(self.file_paths)} InkML files")
        
        # Process data and create label encoder
        self.samples, self.labels, self.label_encoder = self._load_and_process_data(label_encoder)
        
        # Compute class weights for handling imbalanced dataset
        self.class_weights = self._compute_class_weights()
        
        print(f"Loaded {len(self.samples)} valid samples")
        print(f"Number of unique classes: {len(self.label_encoder)}")
    
    def _load_and_process_data(self, label_encoder: Optional[Dict[str, int]]) -> Tuple[List[np.ndarray], List[int], Dict[str, int]]:
        """Load and preprocess all data."""
        
        # Check for cached data
        cache_file = None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)
            cache_file = self.cache_dir / f"processed_data_{len(self.file_paths)}.pkl"
            
            if cache_file.exists():
                print("Loading cached processed data...")
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    return cached_data['samples'], cached_data['labels'], cached_data['label_encoder']
                except Exception as e:
                    print(f"Error loading cache: {e}")
        
        # Process all files
        samples = []
        label_strings = []
        
        print("Processing stroke data...")
        for file_path in tqdm(self.file_paths, desc="Processing files"):
            label, features = self.preprocessor.preprocess_stroke(file_path, augment=False)
            
            if label is not None and features is not None:
                samples.append(features)
                label_strings.append(label)
        
        # Create or use provided label encoder
        if label_encoder is None:
            unique_labels = sorted(set(label_strings))
            label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Convert string labels to integers
        labels = [label_encoder[label] for label in label_strings if label in label_encoder]
        
        # Filter samples to match valid labels
        valid_samples = []
        valid_labels = []
        for sample, label_str in zip(samples, label_strings):
            if label_str in label_encoder:
                valid_samples.append(sample)
                valid_labels.append(label_encoder[label_str])
        
        # Cache processed data
        if cache_file:
            try:
                cache_data = {
                    'samples': valid_samples,
                    'labels': valid_labels,
                    'label_encoder': label_encoder
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"Cached processed data to {cache_file}")
            except Exception as e:
                print(f"Error saving cache: {e}")
        
        return valid_samples, valid_labels, label_encoder
    
    def _compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalanced dataset."""
        class_counts = Counter(self.labels)
        num_classes = len(self.label_encoder)
        
        # Inverse frequency weighting
        weights = torch.zeros(num_classes)
        total_samples = len(self.labels)
        
        for class_idx, count in class_counts.items():
            weights[class_idx] = total_samples / (num_classes * count)
        
        return weights
    
    def get_class_distribution(self) -> pd.DataFrame:
        """Get class distribution statistics."""
        class_counts = Counter(self.labels)
        
        # Create reverse label encoder
        idx_to_label = {idx: label for label, idx in self.label_encoder.items()}
        
        distribution_data = []
        for class_idx, count in class_counts.items():
            label = idx_to_label[class_idx]
            percentage = (count / len(self.labels)) * 100
            distribution_data.append({
                'Symbol': label,
                'Count': count,
                'Percentage': percentage
            })
        
        df = pd.DataFrame(distribution_data)
        return df.sort_values('Count', ascending=False).reset_index(drop=True)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        features = self.samples[idx].copy()  # Copy to avoid modifying original
        label = self.labels[idx]
        
        # Apply augmentation if enabled
        if self.augment:
            features = self.preprocessor.augment_data(features)
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.LongTensor([label])
        
        return features_tensor, label_tensor.squeeze()


class DataManager:
    """Manages data loading, splitting, and normalization."""
    
    def __init__(self,
                 data_dir: Path,
                 target_length: int = 128,
                 feature_dim: int = 8,
                 test_split: float = 0.2,
                 val_split: float = 0.1,
                 cache_dir: Optional[Path] = None,
                 random_seed: int = 42):
        """
        Initialize data manager.
        
        Args:
            data_dir (Path): Directory containing InkML files
            target_length (int): Target sequence length
            feature_dim (int): Feature dimension
            test_split (float): Test set proportion
            val_split (float): Validation set proportion (from remaining after test split)
            cache_dir (Optional[Path]): Cache directory
            random_seed (int): Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.target_length = target_length
        self.feature_dim = feature_dim
        self.test_split = test_split
        self.val_split = val_split
        self.cache_dir = cache_dir
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize preprocessor
        self.preprocessor = StrokePreprocessor(target_length, feature_dim)
        
        # Data splits
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.normalization_stats = None
        
    def prepare_data(self, max_samples: Optional[int] = None) -> Dict[str, StrokeDataset]:
        """
        Prepare train/val/test datasets with proper normalization.
        
        Args:
            max_samples (Optional[int]): Maximum total samples to use
            
        Returns:
            Dict containing train, val, and test datasets
        """
        print("Preparing datasets...")
        
        # Load all data first
        full_dataset = StrokeDataset(
            data_dir=self.data_dir,
            preprocessor=self.preprocessor,
            cache_dir=self.cache_dir,
            max_samples=max_samples,
            augment=False
        )
        
        # Compute normalization statistics on full dataset
        print("Computing normalization statistics...")
        all_features = [sample for sample in full_dataset.samples]
        self.normalization_stats = self.preprocessor.get_data_statistics(all_features)
        
        # Split data
        indices = list(range(len(full_dataset)))
        np.random.shuffle(indices)
        
        # Calculate split sizes
        n_test = int(len(indices) * self.test_split)
        n_val = int(len(indices) * self.val_split)
        n_train = len(indices) - n_test - n_val
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        print(f"Data splits: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
        
        # Create subset datasets
        self.train_dataset = self._create_subset_dataset(full_dataset, train_indices, augment=True)
        self.val_dataset = self._create_subset_dataset(full_dataset, val_indices, augment=False)
        self.test_dataset = self._create_subset_dataset(full_dataset, test_indices, augment=False)
        
        return {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }
    
    def _create_subset_dataset(self, full_dataset: StrokeDataset, indices: List[int], augment: bool) -> StrokeDataset:
        """Create a subset dataset from indices."""
        subset_dataset = StrokeDataset.__new__(StrokeDataset)  # Create without calling __init__
        
        # Copy attributes
        subset_dataset.data_dir = full_dataset.data_dir
        subset_dataset.preprocessor = full_dataset.preprocessor
        subset_dataset.augment = augment
        subset_dataset.cache_dir = full_dataset.cache_dir
        subset_dataset.label_encoder = full_dataset.label_encoder
        
        # Subset the data
        subset_dataset.samples = [full_dataset.samples[i] for i in indices]
        subset_dataset.labels = [full_dataset.labels[i] for i in indices]
        subset_dataset.file_paths = [full_dataset.file_paths[i] for i in indices]
        
        # Recompute class weights for this subset
        subset_dataset.class_weights = subset_dataset._compute_class_weights()
        
        return subset_dataset
    
    def create_data_loaders(self,
                           batch_size: int = 32,
                           num_workers: int = 4,
                           use_weighted_sampling: bool = True) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            batch_size (int): Batch size
            num_workers (int): Number of worker processes
            use_weighted_sampling (bool): Use weighted sampling for training
            
        Returns:
            Dict containing train, val, and test DataLoaders
        """
        if self.train_dataset is None:
            raise ValueError("Must call prepare_data() first")
        
        # Create samplers
        train_sampler = None
        if use_weighted_sampling:
            # Weight samples by inverse class frequency
            sample_weights = []
            for label in self.train_dataset.labels:
                sample_weights.append(self.train_dataset.class_weights[label].item())
            
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        
        # Create DataLoaders
        loaders = {}
        
        loaders['train'] = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_fn
        )
        
        loaders['val'] = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_fn
        )
        
        loaders['test'] = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_fn
        )
        
        return loaders
    
    def _collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom collate function for variable-length sequences.
        
        Args:
            batch: List of (features, label) tuples
            
        Returns:
            Tuple of (padded_features, labels)
        """
        features_list, labels_list = zip(*batch)
        
        # Stack features (they should all be the same length due to preprocessing)
        features_batch = torch.stack(features_list, dim=0)
        labels_batch = torch.stack(labels_list, dim=0)
        
        # Apply normalization using dataset statistics
        if self.normalization_stats:
            mean = torch.FloatTensor(self.normalization_stats['mean'])
            std = torch.FloatTensor(self.normalization_stats['std'])
            features_batch = (features_batch - mean) / std
        
        return features_batch, labels_batch
    
    def get_class_info(self) -> Dict:
        """Get information about classes."""
        if self.train_dataset is None:
            raise ValueError("Must call prepare_data() first")
        
        # Reverse label encoder
        idx_to_label = {idx: label for label, idx in self.train_dataset.label_encoder.items()}
        
        return {
            'num_classes': len(self.train_dataset.label_encoder),
            'label_encoder': self.train_dataset.label_encoder,
            'idx_to_label': idx_to_label,
            'class_weights': self.train_dataset.class_weights
        }
    
    def save_label_encoder(self, path: Path):
        """Save label encoder for later use."""
        if self.train_dataset is None:
            raise ValueError("Must call prepare_data() first")
        
        with open(path, 'wb') as f:
            pickle.dump(self.train_dataset.label_encoder, f)
        print(f"Label encoder saved to {path}")
    
    def load_label_encoder(self, path: Path) -> Dict[str, int]:
        """Load label encoder from file."""
        with open(path, 'rb') as f:
            label_encoder = pickle.load(f)
        return label_encoder


# Example usage and testing
if __name__ == "__main__":
    # Test data loading
    data_dir = Path("symbols")  # Adjust path as needed
    
    if data_dir.exists():
        # Create data manager
        data_manager = DataManager(
            data_dir=data_dir,
            target_length=128,
            feature_dim=8,
            cache_dir=Path("cache")
        )
        
        # Prepare data (use small sample for testing)
        datasets = data_manager.prepare_data(max_samples=100)
        
        # Create data loaders
        loaders = data_manager.create_data_loaders(batch_size=8)
        
        # Test data loading
        train_loader = loaders['train']
        for batch_idx, (features, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Features shape: {features.shape}, Labels shape: {labels.shape}")
            print(f"Features range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"Labels: {labels}")
            
            if batch_idx >= 2:  # Only show first few batches
                break
        
        # Show class distribution
        print("\nClass distribution:")
        class_dist = datasets['train'].get_class_distribution()
        print(class_dist.head(10))
        
        # Show class info
        class_info = data_manager.get_class_info()
        print(f"\nNumber of classes: {class_info['num_classes']}")
        print(f"Class weights range: [{class_info['class_weights'].min():.3f}, {class_info['class_weights'].max():.3f}]")
    
    else:
        print(f"Data directory {data_dir} not found")
import torch
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Sampler
import cv2
import random
import numpy as np


class HYBTr_Dataset(Dataset):

    def __init__(self, params, image_path, label_path, words, is_train=True):
        super(HYBTr_Dataset, self).__init__()
        with open(image_path, 'rb') as f:
            self.images = pkl.load(f)
        with open(label_path, 'rb') as f:
            self.labels = pkl.load(f)

        self.name_list = list(self.labels.keys())
        self.words = words
        self.max_width = params['image_width']
        self.is_train = is_train
        self.params = params
        self.image_height = params['image_height']
        self.image_width = params['image_width']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        name = self.name_list[idx]

        image = self.images[name]

        image = torch.Tensor(image) / 255
        image = image.unsqueeze(0)

        label = self.labels[name]

        child_words = [item.split()[1] for item in label]
        child_words = self.words.encode(child_words)
        child_words = torch.LongTensor(child_words)
        child_ids = [int(item.split()[0]) for item in label]
        child_ids = torch.LongTensor(child_ids)

        parent_words = [item.split()[3] for item in label]
        parent_words = self.words.encode(parent_words)
        parent_words = torch.LongTensor(parent_words)
        parent_ids = [int(item.split()[2]) for item in label]
        parent_ids = torch.LongTensor(parent_ids)


        struct_label = [item.split()[4:] for item in label]
        struct = torch.zeros((len(struct_label), len(struct_label[0]))).long()
        for i in range(len(struct_label)):
            for j in range(len(struct_label[0])):
                struct[i][j] = struct_label[i][j] != 'None'

        label = torch.cat([child_ids.unsqueeze(1), child_words.unsqueeze(1), parent_ids.unsqueeze(1), parent_words.unsqueeze(1), struct], dim=1)

        return image, label

    # batch_images: [Batch Size, (Image, HybridTree)], where
    # - Image = HxW tensor
    # - HybridTree = nr of lines x 11 (id, symbol, parent_id, parent_symbol, ... 7 structs ...)
    def collate_fn(self, batch_images):

        max_width, max_height, max_length = 0, 0, 0
        batch, channel = len(batch_images), batch_images[0][0].shape[0]
        proper_items = []
        for item in batch_images:
            # item = (Image, HybridTree)
            if item[0].shape[1] * max_width > self.image_width * self.image_height or item[0].shape[2] * max_height > self.image_width * self.image_height:
                continue
            max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
            max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
            max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
            proper_items.append(item)

        images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros(
            (len(proper_items), 1, max_height, max_width))
        labels, labels_masks = torch.zeros((len(proper_items), max_length, 11)).long(), \
                               torch.zeros((len(proper_items), max_length, 2))

        # for each item in batch...
        for i in range(len(proper_items)):

            _, h, w = proper_items[i][0].shape
            nr_of_lines = proper_items[i][1].shape[0]

            # image_masks: [batch_index, 0, height, width]
            images[i][:, :h, :w] = proper_items[i][0]
            image_masks[i][:, :h, :w] = 1

            # label_masks: [batch index, line index, (1 if line applies, 1 if line is struct)]
            labels[i][:nr_of_lines, :] = proper_items[i][1]
            labels_masks[i][:nr_of_lines, 0] = 1

            for j in range(nr_of_lines): 
                labels_masks[i][j][1] = proper_items[i][1][j][4:].sum() != 0

        return images, image_masks, labels, labels_masks


class BucketBatchSampler(Sampler):
    """
    Sampler that groups samples by image size (pixels) into buckets with dynamic batch sizes.
    This ensures batches have similar image sizes, leading to:
    - Uniform memory usage (eliminates spikes from large images)
    - Less padding waste
    - Better GPU utilization
    - Adaptive batch sizes: smaller images use larger batches
    """

    def __init__(self, dataset, batch_size, bucket_size_multiplier=10, shuffle=True, use_dynamic_batching=True, params=None):
        """
        Args:
            dataset: HYBTr_Dataset instance
            batch_size: Base number of samples per batch (used for longest sequences)
            bucket_size_multiplier: How many batches to group in one bucket (default: 10)
            shuffle: Whether to shuffle within buckets and across buckets
            use_dynamic_batching: If True, adjust batch_size per bucket to keep tokens/batch constant
            params: Optional params dict for accessing max_batch_size and other config
        """
        self.dataset = dataset
        self.base_batch_size = batch_size
        self.shuffle = shuffle
        self.use_dynamic_batching = use_dynamic_batching
        self.params = params if params is not None else {}

        # Get image sizes for all samples (bucketing by image size instead of sequence length)
        print("BucketBatchSampler: Computing image sizes...")
        self.image_sizes = []  # Now stores (height, width, pixels) tuples
        for idx in range(len(dataset)):
            name = dataset.name_list[idx]
            image = dataset.images[name]
            # Store dimensions and pixels to compute worst-case batch allocation later
            height, width = image.shape[0], image.shape[1]
            img_size = (height, width, height * width)
            self.image_sizes.append(img_size)

        # Validate dataset is not empty
        if len(self.image_sizes) == 0:
            raise ValueError(
                "BucketBatchSampler: Dataset is empty! "
                "Cannot create batches from an empty dataset. "
                "Please check your data paths and ensure dataset contains samples."
            )

        # Sort indices by image size (pixels)
        self.sorted_indices = sorted(range(len(self.image_sizes)), key=lambda i: self.image_sizes[i][2])
        sorted_sizes = [self.image_sizes[idx][2] for idx in self.sorted_indices]  # Extract pixels for display
        if params.get('bucket_trunc_largest') is not None:
            N = params['bucket_trunc_largest']
            print(f"Largest image sizes: {sorted_sizes[-20:]}")
            print(f"Removing largest {N} items")
            del self.sorted_indices[-N:]
            del sorted_sizes[-N:]

        # Create buckets using percentile-based or fixed-size strategy
        if 'bucket_percentiles' in self.params and self.params['bucket_percentiles'] is not None:
            # Percentile-based bucketing: distribute samples by size percentiles
            percentiles = self.params['bucket_percentiles']
            print(f"BucketBatchSampler: Using percentile-based bucketing with percentiles: {percentiles}")

            # Validate percentiles
            if len(percentiles) < 2:
                raise ValueError("bucket_percentiles must have at least 2 values (start and end)")
            if percentiles[0] != 0 or percentiles[-1] != 100:
                raise ValueError("bucket_percentiles must start at 0 and end at 100")
            if percentiles != sorted(percentiles):
                raise ValueError("bucket_percentiles must be in ascending order")

            # Create buckets based on percentile ranges
            n_samples = len(sorted_sizes)
            self.buckets = []
            for i in range(len(percentiles) - 1):
                start_percentile = percentiles[i]
                end_percentile = percentiles[i + 1]
                start_idx = int(n_samples * start_percentile / 100)
                end_idx = int(n_samples * end_percentile / 100)

                bucket = self.sorted_indices[start_idx:end_idx]

                # Validate bucket is not empty (can happen with duplicate/adjacent percentiles)
                if len(bucket) == 0:
                    raise ValueError(
                        f"Bucket {i} is empty! Percentile range [{start_percentile}%, {end_percentile}%) "
                        f"creates no samples. Check percentiles {percentiles} for duplicates or values too close together."
                    )

                self.buckets.append(bucket)
        else:
            # Fixed-size bucketing (backward compatibility)
            print(f"BucketBatchSampler: Using fixed-size bucketing with multiplier={bucket_size_multiplier}")
            bucket_size = batch_size * bucket_size_multiplier
            self.buckets = []
            for i in range(0, len(self.sorted_indices), bucket_size):
                bucket = self.sorted_indices[i:i + bucket_size]
                self.buckets.append(bucket)

        # Calculate dynamic batch sizes per bucket
        if use_dynamic_batching:
            # Target pixels = base_batch_size × max_image_size (from largest images)
            max_size = max(sorted_sizes)
            target_pixels_per_batch = self.base_batch_size * max_size

            # Absolute maximum batch size to prevent OOM (configurable via params)
            absolute_max_batch_size = self.params.get('max_batch_size', 64)  # Default: 64 samples max

            self.bucket_batch_sizes = []
            for bucket in self.buckets:
                image_dims = [self.image_sizes[idx] for idx in bucket]  # List of (h, w, pixels) tuples

                # Calculate worst-case area (max_height × max_width) for actual memory allocation
                max_height = max(dims[0] for dims in image_dims)
                max_width = max(dims[1] for dims in image_dims)
                bucket_worst_case_area = max_height * max_width  # CRITICAL: matches collate_fn allocation

                dynamic_batch_size = max(1, min(target_pixels_per_batch // max(bucket_worst_case_area, 1), absolute_max_batch_size))
                self.bucket_batch_sizes.append(dynamic_batch_size)

            # Print dynamic batching info
            print(f"Dynamic batching enabled: target_pixels_per_batch={target_pixels_per_batch:,} (base_batch_size={self.base_batch_size} × max_size={max_size:,} pixels)")
        else:
            # Fixed batch size for all buckets
            self.bucket_batch_sizes = [self.base_batch_size] * len(self.buckets)
            print(f"BucketBatchSampler: Created {len(self.buckets)} buckets (fixed batch_size={self.base_batch_size})")

        # Show bucket details
        num_buckets = len(self.buckets)

        # For percentile-based bucketing, show all buckets (typically 5-10)
        # For fixed-size bucketing, show representative sample (5 buckets)
        if 'bucket_percentiles' in self.params and self.params['bucket_percentiles'] is not None:
            # Show all buckets for percentile-based
            sample_indices = list(range(num_buckets))
            print(f"\nBucket statistics:")
        else:
            # Show representative buckets for fixed-size
            if num_buckets <= 10:
                sample_indices = list(range(num_buckets))
            else:
                # Sample at 0%, 25%, 50%, 75%, 100% positions
                sample_indices = [
                    0,
                    num_buckets // 4,
                    num_buckets // 2,
                    3 * num_buckets // 4,
                    num_buckets - 1
                ]
            print(f"\nRepresentative bucket statistics (showing {len(sample_indices)} of {num_buckets} buckets):")

        total_nr_batches = 0
        for i in sample_indices:
            bucket = self.buckets[i]
            dims_in_bucket = [self.image_sizes[idx] for idx in bucket]  # (h, w, pixels) tuples
            pixels_in_bucket = [dims[2] for dims in dims_in_bucket]  # Extract pixels for stats

            # Calculate worst-case dimensions
            max_h = max(dims[0] for dims in dims_in_bucket)
            max_w = max(dims[1] for dims in dims_in_bucket)
            worst_case_area = max_h * max_w

            batch_size_for_bucket = self.bucket_batch_sizes[i]
            num_batches = (len(bucket) + batch_size_for_bucket - 1) // batch_size_for_bucket
            total_nr_batches+= num_batches
            avg_pixels = batch_size_for_bucket * np.mean(pixels_in_bucket)
            avg_worst_case = batch_size_for_bucket * worst_case_area

            print(f"  Bucket {i}: {len(bucket):,} samples → ~{num_batches:,} batches, "
                  f"pixel range [{min(pixels_in_bucket):,}, {max(pixels_in_bucket):,}], "
                  f"dims [{max_h}×{max_w}], "
                  f"batch_size={batch_size_for_bucket}, "
                  f"worst_case_pixels/batch={avg_worst_case:,.0f}")
        print(f"Nr of batches: {total_nr_batches}")
            

    def __iter__(self):
        """
        Generate batches by collecting from all buckets and shuffling together.

        Strategy:
        1. For each bucket, shuffle samples and create batches using bucket-specific batch_size
        2. Collect all batches into a single list
        3. Shuffle the entire batch list for diversity across buckets
        4. Yield batches in shuffled order

        This ensures training sees diverse image sizes throughout the epoch (not sequentially by bucket).
        Proportional representation is natural: buckets with more samples contribute more batches.

        Yields:
            list: Batch of sample indices
        """
        all_batches = []

        for bucket, batch_size in zip(self.buckets, self.bucket_batch_sizes):
            # Shuffle samples within bucket
            if self.shuffle:
                bucket_copy = bucket.copy()
                random.shuffle(bucket_copy)
            else:
                bucket_copy = bucket

            # Create batches from this bucket
            for i in range(0, len(bucket_copy), batch_size):
                batch = bucket_copy[i:i + batch_size]
                if len(batch) > 0:
                    all_batches.append(batch)

        # Shuffle all batches together for diversity across buckets
        if self.shuffle:
            random.shuffle(all_batches)

        # Iterate through shuffled batches
        for batch in all_batches:
            yield batch

    def __len__(self):
        """
        Calculate total number of batches across all buckets.

        Uses ceiling division to count partial batches: if a bucket has 35 samples
        and batch_size=32, this counts as 2 batches (32 + 3).

        Returns:
            int: Total number of batches that will be yielded by __iter__
        """
        total_batches = 0
        for bucket, bucket_batch_size in zip(self.buckets, self.bucket_batch_sizes):
            # Count full batches + 1 partial batch if remainder exists (ceiling division)
            total_batches += (len(bucket) + bucket_batch_size - 1) // bucket_batch_size
        return total_batches


def get_dataset(params):

    words = Words(params['word_path'])

    params['word_num'] = len(words)
    params['struct_num'] = 7
    print(f"training data，images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"test data，images: {params['eval_image_path']} labels: {params['eval_label_path']}")
    train_dataset = HYBTr_Dataset(params, params['train_image_path'], params['train_label_path'], words)
    eval_dataset = HYBTr_Dataset(params, params['eval_image_path'], params['eval_label_path'], words)

    # Use bucket batch sampler for training to ensure uniform memory usage
    train_batch_sampler = BucketBatchSampler(
        train_dataset,
        batch_size=params['batch_size'],
        bucket_size_multiplier=params.get('bucket_size_multiplier', 10),
        shuffle=True,
        use_dynamic_batching=params.get('use_dynamic_batching', True),
        params=params  # Pass params for max_batch_size and other config
    )

    # Use bucket batch sampler for eval too (can use larger batches since no gradients/accumulation)
    eval_batch_sampler = BucketBatchSampler(
        eval_dataset,
        batch_size=params.get('eval_batch_size', params['batch_size'] * 2),  # Default: 2× training batch_size
        bucket_size_multiplier=params.get('bucket_size_multiplier', 10),
        shuffle=False,  # Don't shuffle eval for reproducibility
        use_dynamic_batching=params.get('use_dynamic_batching', True),
        params=params
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=params['workers'],
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=False,  # Disabled - was causing CPU oscillation and worker stalls
        prefetch_factor=1 if params['workers'] > 0 else None  # Each worker prefetches 1 batch
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_sampler=eval_batch_sampler,
        num_workers=params['workers'],
        collate_fn=eval_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=False,  # Disabled - was causing CPU oscillation and worker stalls
        prefetch_factor=1 if params['workers'] > 0 else None
    )

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)}')

    return train_loader, eval_loader


class Words:
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'{len(words)} symbols in total')

        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label

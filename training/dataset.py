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

    def collate_fn(self, batch_images):

        max_width, max_height, max_length = 0, 0, 0
        batch, channel = len(batch_images), batch_images[0][0].shape[0]
        proper_items = []
        for item in batch_images:
            if item[0].shape[1] * max_width > self.image_width * self.image_height or item[0].shape[2] * max_height > self.image_width * self.image_height:
                continue
            max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
            max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
            max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
            proper_items.append(item)

        images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros(
            (len(proper_items), 1, max_height, max_width))
        labels, labels_masks = torch.zeros((len(proper_items), max_length, 11)).long(), torch.zeros(
            (len(proper_items), max_length, 2))

        for i in range(len(proper_items)):

            _, h, w = proper_items[i][0].shape
            images[i][:, :h, :w] = proper_items[i][0]
            image_masks[i][:, :h, :w] = 1

            l = proper_items[i][1].shape[0]
            labels[i][:l, :] = proper_items[i][1]
            labels_masks[i][:l, 0] = 1

            for j in range(proper_items[i][1].shape[0]):
                labels_masks[i][j][1] = proper_items[i][1][j][4:].sum() != 0

        return images, image_masks, labels, labels_masks


class BucketBatchSampler(Sampler):
    """
    Sampler that groups samples by sequence length into buckets with dynamic batch sizes.
    This ensures batches have similar sequence lengths, leading to:
    - Uniform memory usage (eliminates spikes from long sequences)
    - Less padding waste
    - Better GPU utilization
    - Adaptive batch sizes: shorter sequences use larger batches
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
        self.lengths = []  # Reusing 'lengths' variable name, but now stores image sizes (pixels)
        for idx in range(len(dataset)):
            name = dataset.name_list[idx]
            image = dataset.images[name]
            # Use total pixels (height × width) as size metric for bucketing
            img_size = image.shape[0] * image.shape[1]
            self.lengths.append(img_size)

        # Validate dataset is not empty
        if len(self.lengths) == 0:
            raise ValueError(
                "BucketBatchSampler: Dataset is empty! "
                "Cannot create batches from an empty dataset. "
                "Please check your data paths and ensure dataset contains samples."
            )

        # Sort indices by image size (pixels)
        self.sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        bucket_size = batch_size * bucket_size_multiplier
        self.buckets = []
        for i in range(0, len(self.sorted_indices), bucket_size):
            bucket = self.sorted_indices[i:i + bucket_size]
            self.buckets.append(bucket)

        # Calculate dynamic batch sizes per bucket
        if use_dynamic_batching:
            # Target pixels = base_batch_size × max_image_size (from largest images)
            max_size = max(self.lengths)
            target_pixels_per_batch = self.base_batch_size * max_size

            # Absolute maximum batch size to prevent OOM (configurable via params)
            absolute_max_batch_size = self.params.get('max_batch_size', 64)  # Default: 64 samples max

            self.bucket_batch_sizes = []
            for bucket in self.buckets:
                image_sizes = [self.lengths[idx] for idx in bucket]  # Image sizes (pixels)

                bucket_max_image_size = max(image_sizes)  # Max image pixels in this bucket
                dynamic_batch_size = max(1, min(target_pixels_per_batch // max(bucket_max_image_size, 1), absolute_max_batch_size))
                self.bucket_batch_sizes.append(dynamic_batch_size)

            # Print dynamic batching info
            print(f"BucketBatchSampler: Created {len(self.buckets)} buckets")
            print(f"Dynamic batching enabled: target_pixels_per_batch={target_pixels_per_batch:,} (base_batch_size={self.base_batch_size} × max_size={max_size:,} pixels)")
        else:
            # Fixed batch size for all buckets
            self.bucket_batch_sizes = [self.base_batch_size] * len(self.buckets)
            print(f"BucketBatchSampler: Created {len(self.buckets)} buckets (fixed batch_size={self.base_batch_size})")

        # Show representative buckets from small to large (5 samples distributed across range)
        num_buckets = len(self.buckets)
        if num_buckets <= 5:
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

        for i in sample_indices:
            bucket = self.buckets[i]
            sizes_in_bucket = [self.lengths[idx] for idx in bucket]
            batch_size_for_bucket = self.bucket_batch_sizes[i]
            avg_pixels = batch_size_for_bucket * np.mean(sizes_in_bucket)
            print(f"  Bucket {i}: {len(bucket)} samples, "
                  f"size range (pixels) [{min(sizes_in_bucket):,}, {max(sizes_in_bucket):,}], "
                  f"mean={np.mean(sizes_in_bucket):,.0f}, "
                  f"batch_size={batch_size_for_bucket}, "
                  f"avg_pixels/batch={avg_pixels:,.0f}")

    def __iter__(self):
        # Create list of (bucket, batch_size) pairs for shuffling
        bucket_pairs = list(zip(self.buckets, self.bucket_batch_sizes))

        # Shuffle buckets for randomness across epochs
        if self.shuffle:
            random.shuffle(bucket_pairs)

        # Iterate through buckets and create batches
        for bucket, bucket_batch_size in bucket_pairs:
            # Shuffle within bucket
            if self.shuffle:
                bucket_copy = bucket.copy()
                random.shuffle(bucket_copy)
            else:
                bucket_copy = bucket

            # Create batches from this bucket using bucket-specific batch size
            for i in range(0, len(bucket_copy), bucket_batch_size):
                batch = bucket_copy[i:i + bucket_batch_size]
                if len(batch) > 0:
                    yield batch

    def __len__(self):
        # Calculate total number of batches using bucket-specific batch sizes
        total_batches = 0
        for bucket, bucket_batch_size in zip(self.buckets, self.bucket_batch_sizes):
            # Count full batches + 1 partial batch if remainder exists
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

    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                              num_workers=params['workers'], collate_fn=train_dataset.collate_fn, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_sampler=eval_batch_sampler,
                              num_workers=params['workers'], collate_fn=eval_dataset.collate_fn, pin_memory=True)

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

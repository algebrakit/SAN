import os
import glob
import xml.etree.ElementTree as ET
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import pickle as pkl
import hashlib
import json
from latex_parser import LaTeXToSANConverter


class StrokeNormalizer:
    """Expression-level stroke normalization for mathematical expressions"""
    
    def __init__(self, params):
        # Target coordinate ranges (preserving mathematical aspect ratios)
        self.target_width = params.get('stroke_target_width', 4.0)   # [-2, 2]  
        self.target_height = params.get('stroke_target_height', 1.0) # [-0.5, 0.5]
        self.min_char_size = params.get('min_char_size', 0.05)       # Prevent tiny symbols
        self.max_aspect_ratio = params.get('max_aspect_ratio', 10.0)  # Limit extreme ratios
        
        # Sequence normalization parameters
        self.max_strokes = params.get('max_strokes', 20)
        self.max_points_per_stroke = params.get('max_points_per_stroke', 100)
    
    def parse_inkml(self, file_path):
        """Parse .inkml file and extract stroke data"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract namespace
            namespace = {'ink': 'http://www.w3.org/2003/InkML'}
            
            # Get labels
            label_elem = root.find('.//ink:annotation[@type="label"]', namespace)
            normalized_label_elem = root.find('.//ink:annotation[@type="normalizedLabel"]', namespace)
            
            # Use normalized label if available, otherwise use regular label
            if normalized_label_elem is not None and normalized_label_elem.text:
                label = normalized_label_elem.text.strip()
            elif label_elem is not None and label_elem.text:
                label = label_elem.text.strip()
            else:
                print(f"Warning: No label found in {file_path}")
                return None
            
            # Extract all traces (strokes)
            strokes = []
            traces = root.findall('.//ink:trace', namespace)
            
            for trace in traces:
                trace_id = trace.get('id')
                trace_data = trace.text.strip() if trace.text else ""
                
                if not trace_data:
                    continue
                
                # Parse points: "x y t,x y t,x y t"
                points = []
                for point_str in trace_data.split(','):
                    coords = point_str.strip().split()
                    if len(coords) >= 3:
                        try:
                            x, y, t = float(coords[0]), float(coords[1]), float(coords[2])
                            points.append((x, y, t))
                        except ValueError:
                            continue
                
                if points:  # Only add strokes with valid points
                    strokes.append({
                        'id': trace_id,
                        'points': points
                    })
            
            return {
                'label': label,
                'strokes': strokes,
                'file_path': file_path
            }
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def normalize_expression_coordinates(self, strokes):
        """Normalize entire expression coordinates while preserving relationships"""
        if not strokes or not any(stroke['points'] for stroke in strokes):
            return strokes, {'scale_factor': 1.0, 'offset': (0, 0)}
        
        # Extract all coordinates from entire expression
        all_points = []
        for stroke in strokes:
            all_points.extend(stroke['points'])
        
        if not all_points:
            return strokes, {'scale_factor': 1.0, 'offset': (0, 0)}
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        # Expression bounding box
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        expr_width = x_max - x_min if x_max != x_min else 1.0
        expr_height = y_max - y_min if y_max != y_min else 1.0
        expr_aspect = expr_width / expr_height
        
        # Handle extreme aspect ratios
        if expr_aspect > self.max_aspect_ratio:
            # Very long expressions: allow some height compression
            target_aspect = self.max_aspect_ratio
            effective_target_height = self.target_width / target_aspect
        else:
            # Normal expressions: use standard target dimensions
            effective_target_height = self.target_height
        
        # Calculate scale factors
        scale_x = self.target_width / expr_width
        scale_y = effective_target_height / expr_height
        
        # Use uniform scaling to preserve aspect ratio
        scale = min(scale_x, scale_y)
        
        # Ensure minimum character size
        estimated_char_width = expr_width / max(len(strokes), 1)
        scaled_char_width = estimated_char_width * scale
        
        if scaled_char_width < self.min_char_size:
            scale = self.min_char_size / estimated_char_width
        
        # Center the expression in target coordinate space
        scaled_width = expr_width * scale
        scaled_height = expr_height * scale
        
        offset_x = -x_min * scale - scaled_width / 2
        offset_y = -y_min * scale - scaled_height / 2
        
        # Apply normalization to all strokes
        normalized_strokes = []
        for stroke in strokes:
            normalized_points = []
            for x, y, t in stroke['points']:
                norm_x = x * scale + offset_x
                norm_y = y * scale + offset_y
                normalized_points.append((norm_x, norm_y, t))
            
            normalized_strokes.append({
                'id': stroke['id'],
                'points': normalized_points
            })
        
        norm_info = {
            'scale_factor': scale,
            'offset': (offset_x, offset_y),
            'original_bounds': (x_min, y_min, x_max, y_max),
            'aspect_ratio': expr_aspect,
            'estimated_char_size': scaled_char_width
        }
        
        return normalized_strokes, norm_info
    
    def normalize_temporal_sequence(self, strokes):
        """Normalize timing while preserving stroke order"""
        if not strokes:
            return strokes
        
        # Extract all timestamps and stroke start times
        all_timestamps = []
        stroke_start_times = []
        
        for stroke in strokes:
            if stroke['points']:
                stroke_times = [p[2] for p in stroke['points']]
                stroke_start_times.append(min(stroke_times))
                all_timestamps.extend(stroke_times)
        
        if not all_timestamps:
            return strokes
        
        # Global time normalization
        t_min, t_max = min(all_timestamps), max(all_timestamps)
        total_duration = t_max - t_min if t_max != t_min else 1.0
        
        # Stroke ordering (important for mathematical structure)
        stroke_order = sorted(range(len(stroke_start_times)), 
                             key=lambda i: stroke_start_times[i])
        
        normalized_strokes = []
        for stroke_idx, stroke in enumerate(strokes):
            if not stroke['points']:
                normalized_strokes.append(stroke)
                continue
            
            normalized_points = []
            for point_idx, (x, y, t) in enumerate(stroke['points']):
                # Global temporal position [0, 1]
                global_t = (t - t_min) / total_duration
                
                # Stroke order encoding
                stroke_order_norm = stroke_order.index(stroke_idx) / (len(strokes) - 1) if len(strokes) > 1 else 0.0
                
                # Point position within stroke [0, 1] 
                point_progress = point_idx / (len(stroke['points']) - 1) if len(stroke['points']) > 1 else 0.0
                
                # Combined temporal encoding
                enhanced_t = (global_t * 0.6 + stroke_order_norm * 0.25 + point_progress * 0.15)
                
                normalized_points.append((x, y, enhanced_t))
            
            normalized_strokes.append({
                'id': stroke['id'],
                'points': normalized_points
            })
        
        return normalized_strokes
    
    def pad_and_truncate_sequences(self, strokes):
        """Handle variable sequence lengths for batching"""
        # Truncate/pad stroke count
        processed_strokes = strokes[:self.max_strokes]
        while len(processed_strokes) < self.max_strokes:
            processed_strokes.append({'id': f'pad_{len(processed_strokes)}', 'points': []})
        
        # Truncate/pad points per stroke
        final_strokes = []
        for stroke in processed_strokes:
            points = stroke['points'][:self.max_points_per_stroke]
            
            # Pad with zeros (will be masked out)
            while len(points) < self.max_points_per_stroke:
                points.append((0.0, 0.0, 0.0))
            
            final_strokes.append({
                'id': stroke['id'],
                'points': points
            })
        
        return final_strokes
    
    def generate_stroke_metadata(self, padded_strokes):
        """Generate masks and positional information for each stroke"""
        stroke_masks = []
        stroke_positions = []
        stroke_bounds = []
        
        for stroke in padded_strokes:
            # Point-level mask
            point_mask = []
            valid_points = []
            
            for x, y, t in stroke['points']:
                if x != 0.0 or y != 0.0 or t != 0.0:  # Valid point
                    point_mask.append(1.0)
                    valid_points.append((x, y))
                else:
                    point_mask.append(0.0)  # Padding point
            
            stroke_masks.append(point_mask)
            
            # Stroke bounding box and position
            if valid_points:
                xs, ys = zip(*valid_points)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                # Centroid for spatial positioning
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                stroke_positions.append([center_x, center_y, width, height])
                stroke_bounds.append([x_min, y_min, x_max, y_max])
            else:
                stroke_positions.append([0.0, 0.0, 0.0, 0.0])
                stroke_bounds.append([0.0, 0.0, 0.0, 0.0])
        
        return (torch.tensor(stroke_masks, dtype=torch.float32),
                torch.tensor(stroke_positions, dtype=torch.float32),
                torch.tensor(stroke_bounds, dtype=torch.float32))
    
    def normalize_expression(self, raw_strokes):
        """Complete expression-level normalization pipeline"""
        # Stage 1: Spatial normalization (entire expression)
        spatial_normalized, spatial_info = self.normalize_expression_coordinates(raw_strokes)
        
        # Stage 2: Temporal normalization (preserve stroke order)
        temporal_normalized = self.normalize_temporal_sequence(spatial_normalized)
        
        # Stage 3: Sequence padding/truncation
        padded_strokes = self.pad_and_truncate_sequences(temporal_normalized)
        
        # Stage 4: Generate masks and auxiliary data
        stroke_masks, stroke_positions, stroke_bounds = self.generate_stroke_metadata(padded_strokes)
        
        return {
            'normalized_strokes': padded_strokes,        # [max_strokes, max_points, 3]
            'stroke_masks': stroke_masks,                # [max_strokes, max_points]
            'stroke_positions': stroke_positions,        # [max_strokes, 4]
            'stroke_bounds': stroke_bounds,              # [max_strokes, 4]
            'expression_metadata': spatial_info,         # Global normalization info
            'num_valid_strokes': len([s for s in raw_strokes if s['points']])
        }


class InkMLDataset(Dataset):
    """Dataset for loading InkML files for stroke-aware SAN training"""
    
    def __init__(self, params, inkml_folder, words, is_train=True):
        super(InkMLDataset, self).__init__()
        
        self.params = params
        self.is_train = is_train
        self.words = words
        self.normalizer = StrokeNormalizer(params)
        self.inkml_folder = inkml_folder
        
        # Store parameters for LaTeX converter
        self.word_path = params.get('word_path', 'data/word.txt')
        self.max_length = params.get('max_sequence_length', 50)
        
        # Create cache directory
        cache_dir = params.get('cache_dir', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check cache settings
        use_cache = params.get('use_cache', True)
        force_rebuild = params.get('force_rebuild_cache', False)
        
        # Try to load from cache first (unless disabled or force rebuild)
        cache_loaded = False
        if use_cache and not force_rebuild:
            cache_loaded = self._load_from_cache()
            if cache_loaded:
                print(f"Loaded {len(self.data_samples)} samples from cache")
        
        if not cache_loaded:
            # Parse InkML files and save to cache
            self._parse_and_cache_data()
            if not use_cache:
                print("Note: Caching disabled - data will be re-parsed next time")
        
        # Extract unique symbols for vocabulary
        self._build_vocabulary()
    
    def _generate_cache_key(self):
        """Generate unique cache key based on dataset parameters"""
        cache_params = {
            'inkml_folder': self.inkml_folder,
            'stroke_target_width': self.params.get('stroke_target_width', 4.0),
            'stroke_target_height': self.params.get('stroke_target_height', 1.0),
            'max_strokes': self.params.get('max_strokes', 20),
            'max_points_per_stroke': self.params.get('max_points_per_stroke', 100),
            'min_char_size': self.params.get('min_char_size', 0.05),
            'max_aspect_ratio': self.params.get('max_aspect_ratio', 10.0)
        }
        
        # Add file count and modification times for cache invalidation
        inkml_files = glob.glob(os.path.join(self.inkml_folder, "*.inkml"))
        cache_params['file_count'] = len(inkml_files)
        
        # Sample a few files for modification time check (to detect dataset changes)
        if inkml_files:
            sample_files = inkml_files[:min(10, len(inkml_files))]
            cache_params['sample_mtimes'] = [os.path.getmtime(f) for f in sample_files]
        
        # Create hash
        cache_str = json.dumps(cache_params, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cache_path(self):
        """Get cache file path"""
        cache_key = self._generate_cache_key()
        cache_dir = self.params.get('cache_dir', 'cache')
        return os.path.join(cache_dir, f'inkml_dataset_{cache_key}.pkl')
    
    def _load_from_cache(self):
        """Try to load dataset from cache (handles both regular and chunked caches)"""
        cache_path = self._get_cache_path()
        metadata_path = f"{cache_path}.meta"
        
        # Check if chunked cache exists
        if os.path.exists(metadata_path):
            return self._load_chunked_cache(cache_path, metadata_path)
        elif os.path.exists(cache_path):
            return self._load_regular_cache(cache_path)
        else:
            print(f"No cache found at {cache_path}")
            return False
    
    def _load_regular_cache(self, cache_path):
        """Load regular (non-chunked) cache"""
        try:
            file_size = os.path.getsize(cache_path) / (1024 * 1024)  # MB
            print(f"Loading dataset from cache: {cache_path} ({file_size:.1f} MB)")
            
            with open(cache_path, 'rb') as f:
                cache_data = pkl.load(f)
            
            self.data_samples = cache_data['data_samples']
            self.inkml_files = cache_data['inkml_files']
            
            print(f"Cache loaded successfully: {len(self.data_samples)} samples")
            return True
            
        except Exception as e:
            print(f"Failed to load regular cache: {e}")
            self._cleanup_corrupted_cache(cache_path)
            return False
    
    def _load_chunked_cache(self, cache_path, metadata_path):
        """Load chunked cache"""
        try:
            print(f"Loading chunked dataset from cache...")
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pkl.load(f)
            
            total_samples = metadata['total_samples']
            num_chunks = metadata['num_chunks']
            
            print(f"Found chunked cache: {total_samples:,} samples in {num_chunks} chunks")
            
            # Check if we should load only one chunk (for reduced memory usage)
            load_single_chunk = self.params.get('load_single_chunk', False)
            single_chunk_index = self.params.get('single_chunk_index', 0)
            
            if load_single_chunk:
                print(f"Loading only chunk {single_chunk_index} to reduce memory usage...")
                chunk_path = f"{cache_path}.chunk{single_chunk_index}"
                
                if not os.path.exists(chunk_path):
                    print(f"Missing chunk file: {chunk_path}")
                    return False
                
                with open(chunk_path, 'rb') as f:
                    chunk_data = pkl.load(f)
                
                self.data_samples = chunk_data
                self.inkml_files = metadata['inkml_files']
                
                print(f"Single chunk loaded successfully: {len(self.data_samples):,} samples from chunk {single_chunk_index}")
                return True
            
            # Load chunks
            self.data_samples = []
            self.inkml_files = metadata['inkml_files']
            
            for chunk_idx in range(num_chunks):
                chunk_path = f"{cache_path}.chunk{chunk_idx}"
                
                if not os.path.exists(chunk_path):
                    print(f"Missing chunk file: {chunk_path}")
                    return False
                
                with open(chunk_path, 'rb') as f:
                    chunk_data = pkl.load(f)
                
                self.data_samples.extend(chunk_data)
                
                if chunk_idx % 5 == 0 or chunk_idx == num_chunks - 1:
                    print(f"  Loaded chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_data):,} samples)")
            
            print(f"Chunked cache loaded successfully: {len(self.data_samples):,} samples")
            return True
            
        except Exception as e:
            print(f"Failed to load chunked cache: {e}")
            import traceback
            traceback.print_exc()
            self._cleanup_corrupted_cache(cache_path, chunked=True)
            return False
    
    def _cleanup_corrupted_cache(self, cache_path, chunked=False):
        """Clean up corrupted cache files"""
        files_to_remove = [cache_path]
        
        if chunked:
            # Remove metadata and all chunk files
            metadata_path = f"{cache_path}.meta"
            if os.path.exists(metadata_path):
                files_to_remove.append(metadata_path)
            
            # Find all chunk files
            cache_dir = os.path.dirname(cache_path)
            cache_name = os.path.basename(cache_path)
            for file in os.listdir(cache_dir):
                if file.startswith(cache_name) and '.chunk' in file:
                    files_to_remove.append(os.path.join(cache_dir, file))
        
        for file_path in files_to_remove:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed corrupted cache file: {file_path}")
            except:
                pass
    
    def _save_to_cache(self):
        """Save dataset to cache using chunked approach"""
        cache_path = self._get_cache_path()
        
        # For very large datasets, use chunked caching
        chunk_size = 50000  # Process in chunks of 50k samples
        total_samples = len(self.data_samples)
        
        if total_samples > 100000:
            print(f"Large dataset detected ({total_samples:,} samples). Using chunked caching...")
            self._save_chunked_cache(cache_path, chunk_size)
        else:
            # Small datasets - use regular caching
            self._save_regular_cache(cache_path)
    
    def _save_regular_cache(self, cache_path):
        """Save dataset using regular approach (for smaller datasets)"""
        cache_data = {
            'data_samples': self.data_samples,
            'inkml_files': self.inkml_files,
            'chunk_info': {'chunked': False, 'total_samples': len(self.data_samples)},
            'params_used': {
                'stroke_target_width': self.params.get('stroke_target_width', 4.0),
                'stroke_target_height': self.params.get('stroke_target_height', 1.0),
                'max_strokes': self.params.get('max_strokes', 20),
                'max_points_per_stroke': self.params.get('max_points_per_stroke', 100)
            }
        }
        
        try:
            print(f"Saving {len(self.data_samples)} samples to cache...")
            with open(cache_path, 'wb') as f:
                pkl.dump(cache_data, f, protocol=pkl.HIGHEST_PROTOCOL)
            
            file_size = os.path.getsize(cache_path) / (1024 * 1024)
            print(f"Dataset cached to: {cache_path}")
            print(f"Cache file size: {file_size:.1f} MB")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_chunked_cache(self, cache_path, chunk_size):
        """Save large dataset using chunked approach"""
        cache_dir = os.path.dirname(cache_path)
        cache_name = os.path.splitext(os.path.basename(cache_path))[0]
        
        try:
            total_samples = len(self.data_samples)
            num_chunks = (total_samples + chunk_size - 1) // chunk_size
            
            print(f"Saving {total_samples:,} samples in {num_chunks} chunks of {chunk_size:,}...")
            
            # Save metadata
            metadata = {
                'chunked': True,
                'total_samples': total_samples,
                'chunk_size': chunk_size,
                'num_chunks': num_chunks,
                'inkml_files': self.inkml_files,
                'params_used': {
                    'stroke_target_width': self.params.get('stroke_target_width', 4.0),
                    'stroke_target_height': self.params.get('stroke_target_height', 1.0),
                    'max_strokes': self.params.get('max_strokes', 20),
                    'max_points_per_stroke': self.params.get('max_points_per_stroke', 100)
                }
            }
            
            metadata_path = f"{cache_path}.meta"
            with open(metadata_path, 'wb') as f:
                pkl.dump(metadata, f, protocol=pkl.HIGHEST_PROTOCOL)
            
            # Save chunks
            total_size = 0
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, total_samples)
                chunk_data = self.data_samples[start_idx:end_idx]
                
                chunk_path = f"{cache_path}.chunk{chunk_idx}"
                with open(chunk_path, 'wb') as f:
                    pkl.dump(chunk_data, f, protocol=pkl.HIGHEST_PROTOCOL)
                
                chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
                total_size += chunk_size_mb
                
                if chunk_idx % 5 == 0 or chunk_idx == num_chunks - 1:
                    print(f"  Saved chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_data):,} samples, {chunk_size_mb:.1f} MB)")
            
            print(f"Dataset cached in {num_chunks} chunks")
            print(f"Total cache size: {total_size:.1f} MB")
            
        except Exception as e:
            print(f"Warning: Failed to save chunked cache: {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_and_cache_data(self):
        """Parse InkML files and save to cache"""
        # Find all InkML files
        self.inkml_files = glob.glob(os.path.join(self.inkml_folder, "*.inkml"))
        
        # Parse and load labels
        self.data_samples = []
        print(f"Parsing {len(self.inkml_files)} InkML files from {self.inkml_folder}...")
        print("This will take several minutes but will be cached for future runs.")
        
        failed_count = 0
        for i, file_path in enumerate(self.inkml_files):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(self.inkml_files)} files...")
            
            parsed_data = self.normalizer.parse_inkml(file_path)
            if parsed_data is not None:
                self.data_samples.append(parsed_data)
            else:
                failed_count += 1
        
        print(f"Successfully parsed {len(self.data_samples)} InkML files")
        if failed_count > 0:
            print(f"Failed to parse {failed_count} files")
        
        # Save to cache (if caching is enabled)
        if self.params.get('use_cache', True):
            self._save_to_cache()
        else:
            print("Caching disabled - parsed data will not be saved")
    
    def _build_vocabulary(self):
        """Build vocabulary from parsed labels - placeholder for now"""
        print("Note: Using existing word vocabulary from word.txt")
        # The words object should already be loaded from word.txt
        # We'll use the existing SAN vocabulary system
    
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        # Normalize strokes
        normalized_data = self.normalizer.normalize_expression(sample['strokes'])
        
        # Convert strokes to tensor format
        stroke_tensor = torch.zeros(self.normalizer.max_strokes, self.normalizer.max_points_per_stroke, 3)
        
        for stroke_idx, stroke in enumerate(normalized_data['normalized_strokes']):
            for point_idx, (x, y, t) in enumerate(stroke['points']):
                stroke_tensor[stroke_idx, point_idx, 0] = x
                stroke_tensor[stroke_idx, point_idx, 1] = y  
                stroke_tensor[stroke_idx, point_idx, 2] = t
        
        # Convert LaTeX label to SAN hybrid tree format
        label_text = sample['label']
        
        # Use the LaTeX converter to create real labels
        if not hasattr(self, 'latex_converter'):
            # Initialize converter with vocabulary path
            word_path = getattr(self, 'word_path', 'data/word.txt')
            max_length = getattr(self, 'max_length', 50)
            self.latex_converter = LaTeXToSANConverter(word_path, max_length)
        
        # Convert LaTeX to SAN label tensor [max_length, 11]
        real_labels = self.latex_converter.convert(label_text)
        
        return stroke_tensor, normalized_data['stroke_masks'], normalized_data['stroke_positions'], real_labels
    
    def collate_fn(self, batch):
        """Collate function for batching stroke data"""
        batch_strokes = []
        batch_stroke_masks = []
        batch_positions = []
        batch_labels = []
        
        for stroke_tensor, stroke_masks, stroke_positions, labels in batch:
            batch_strokes.append(stroke_tensor)
            batch_stroke_masks.append(stroke_masks)
            batch_positions.append(stroke_positions)
            batch_labels.append(labels)
        
        # Stack tensors - all should be same size due to padding
        strokes_batch = torch.stack(batch_strokes)          # [batch, max_strokes, max_points, 3]
        masks_batch = torch.stack(batch_stroke_masks)       # [batch, max_strokes, max_points]
        positions_batch = torch.stack(batch_positions)      # [batch, max_strokes, 4]
        labels_batch = torch.stack(batch_labels)            # [batch, max_length, 11]
        
        # Create label masks based on actual content
        # label_masks[:, :, 0] = token validity mask (1 if token exists, 0 if padding)
        # label_masks[:, :, 1] = structure validity mask (1 if structure prediction needed)
        batch_size, max_length = labels_batch.shape[:2]
        label_masks = torch.zeros(batch_size, max_length, 2)
        
        for batch_idx in range(batch_size):
            # Find valid tokens (non-zero token IDs, excluding padding)
            valid_tokens = labels_batch[batch_idx, :, 1] > 0
            
            # Token validity mask
            label_masks[batch_idx, :, 0] = valid_tokens.float()
            
            # Structure validity mask (for structure predictions)
            # Only predict structure for non-padding tokens
            label_masks[batch_idx, :, 1] = valid_tokens.float()
        
        return strokes_batch, masks_batch, positions_batch, labels_batch, label_masks


def get_stroke_dataset(params):
    """Create stroke-aware datasets for training"""
    
    # Load word vocabulary 
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['struct_num'] = 7  # Keep same as original
    
    # Set stroke-specific parameters
    if 'stroke_target_width' not in params:
        params['stroke_target_width'] = 4.0
    if 'stroke_target_height' not in params:
        params['stroke_target_height'] = 1.0
    if 'max_strokes' not in params:
        params['max_strokes'] = 20
    if 'max_points_per_stroke' not in params:
        params['max_points_per_stroke'] = 100
    
    print(f"Stroke dataset parameters:")
    print(f"  Target dimensions: {params['stroke_target_width']} × {params['stroke_target_height']}")
    print(f"  Max strokes per expression: {params['max_strokes']}")
    print(f"  Max points per stroke: {params['max_points_per_stroke']}")
    
    # Create datasets
    inkml_folder = params.get('inkml_folder', 'synthetic')
    train_dataset = InkMLDataset(params, inkml_folder, words, is_train=True)
    
    # For now, use same dataset for eval (should split later)
    eval_dataset = InkMLDataset(params, inkml_folder, words, is_train=False)
    
    # Create samplers
    train_sampler = RandomSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset)  # Use sequential for eval
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=train_dataset.collate_fn, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=eval_dataset.collate_fn, pin_memory=True)
    
    print(f'Stroke train dataset: {len(train_dataset)} samples, {len(train_loader)} batches')
    print(f'Stroke eval dataset: {len(eval_dataset)} samples, {len(eval_loader)} batches')
    
    return train_loader, eval_loader


class Words:
    """Word vocabulary class - reuse from original dataset.py"""
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'{len(words)} symbols in vocabulary')

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
"""
Stroke Preprocessing Module

This module handles spatial and temporal normalization of stroke data
for mathematical symbol recognition.
"""

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')


class StrokePreprocessor:
    """Preprocesses stroke data with spatial and temporal normalization."""
    
    def __init__(self, target_length: int = 128, feature_dim: int = 8):
        """
        Initialize the preprocessor.
        
        Args:
            target_length (int): Target sequence length after resampling
            feature_dim (int): Number of features per point (x, y, vx, vy, speed, direction, curvature, pen_state)
        """
        self.target_length = target_length
        self.feature_dim = feature_dim
    
    def parse_inkml_strokes(self, file_path: Path) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Parse InkML file and extract strokes and label.
        
        Args:
            file_path (Path): Path to InkML file
            
        Returns:
            Tuple[Optional[str], Optional[np.ndarray]]: (label, strokes_array)
            strokes_array shape: (n_points, 3) where columns are [x, y, stroke_id]
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract label
            label = None
            for annotation in root.findall('.//{http://www.w3.org/2003/InkML}annotation'):
                if annotation.get('type') == 'label':
                    label = annotation.text
                    break
            
            # Extract strokes
            strokes = []
            stroke_id = 0
            for trace in root.findall('.//{http://www.w3.org/2003/InkML}trace'):
                if trace.text:
                    # Parse coordinate pairs
                    points = []
                    coord_text = trace.text.strip()
                    
                    # Handle different coordinate formats
                    if ',' in coord_text:
                        # Format: "x1 y1 timestamp1, x2 y2 timestamp2, ..." or "x1 y1, x2 y2, ..."
                        coord_pairs = coord_text.split(',')
                        for pair in coord_pairs:
                            coords = pair.split()
                            if len(coords) >= 2:
                                x, y = float(coords[0]), float(coords[1])
                                # Ignore timestamp if present (coords[2])
                                points.append([float(x), float(y), stroke_id])
                    else:
                        # Format: "x1 y1 x2 y2 ..." (space separated)
                        coords = coord_text.split()
                        # Handle both 2-value (x y) and 3-value (x y timestamp) formats
                        if len(coords) >= 2:
                            step = 3 if len(coords) % 3 == 0 and len(coords) >= 3 else 2
                            for i in range(0, len(coords)-1, step):
                                if i+1 < len(coords):
                                    x, y = float(coords[i]), float(coords[i+1])
                                    points.append([x, y, stroke_id])
                    
                    if points:
                        strokes.extend(points)
                        stroke_id += 1
            
            if not strokes:
                return label, None
                
            return label, np.array(strokes)
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None, None
    
    def spatial_normalization(self, strokes: np.ndarray) -> np.ndarray:
        """
        Apply spatial normalization: centering, scaling, and coordinate system standardization.
        
        Args:
            strokes (np.ndarray): Raw stroke data [x, y, stroke_id]
            
        Returns:
            np.ndarray: Spatially normalized strokes
        """
        if len(strokes) == 0:
            return strokes
            
        normalized = strokes.copy()
        
        # Extract coordinates
        x_coords = normalized[:, 0]
        y_coords = normalized[:, 1]
        
        # Center to origin (translation invariance)
        x_mean, y_mean = np.mean(x_coords), np.mean(y_coords)
        normalized[:, 0] -= x_mean
        normalized[:, 1] -= y_mean
        
        # Scale to unit bounding box (size invariance)
        x_range = np.max(normalized[:, 0]) - np.min(normalized[:, 0])
        y_range = np.max(normalized[:, 1]) - np.min(normalized[:, 1])
        
        if x_range > 0 or y_range > 0:
            scale_factor = max(x_range, y_range, 1e-6)  # Avoid division by zero
            normalized[:, 0] /= scale_factor
            normalized[:, 1] /= scale_factor
        
        return normalized
    
    def temporal_resampling(self, strokes: np.ndarray) -> np.ndarray:
        """
        Resample strokes to consistent temporal spacing.
        
        Args:
            strokes (np.ndarray): Stroke data [x, y, stroke_id]
            
        Returns:
            np.ndarray: Resampled stroke data
        """
        if len(strokes) <= 2:
            # Pad very short strokes
            padded = np.zeros((self.target_length, 3))
            padded[:len(strokes)] = strokes
            return padded
        
        # Compute cumulative arc length for natural parameterization
        distances = np.sqrt(np.diff(strokes[:, 0])**2 + np.diff(strokes[:, 1])**2)
        distances = np.concatenate([[0], distances])
        cumulative_distances = np.cumsum(distances)
        
        if cumulative_distances[-1] == 0:
            # All points are the same - just repeat
            resampled = np.tile(strokes[0], (self.target_length, 1))
            return resampled
        
        # Create uniform sampling points
        total_length = cumulative_distances[-1]
        uniform_distances = np.linspace(0, total_length, self.target_length)
        
        # Interpolate coordinates at uniform distances
        try:
            interp_x = interpolate.interp1d(cumulative_distances, strokes[:, 0], 
                                          kind='linear', fill_value='extrapolate')
            interp_y = interpolate.interp1d(cumulative_distances, strokes[:, 1], 
                                          kind='linear', fill_value='extrapolate')
            interp_stroke_id = interpolate.interp1d(cumulative_distances, strokes[:, 2], 
                                                  kind='nearest', fill_value='extrapolate')
            
            # Ensure exact target length
            x_interp = interp_x(uniform_distances)
            y_interp = interp_y(uniform_distances)
            stroke_id_interp = interp_stroke_id(uniform_distances)
            
            # Force exact shape match
            resampled = np.zeros((self.target_length, 3))
            resampled[:, 0] = x_interp[:self.target_length]
            resampled[:, 1] = y_interp[:self.target_length]
            resampled[:, 2] = stroke_id_interp[:self.target_length]
            
        except Exception as e:
            # Fallback: simple linear interpolation
            resampled = np.zeros((self.target_length, 3))
            indices = np.linspace(0, len(strokes)-1, self.target_length)
            
            for i, idx in enumerate(indices):
                if idx == int(idx):
                    resampled[i] = strokes[int(idx)]
                else:
                    # Linear interpolation
                    idx_low, idx_high = int(np.floor(idx)), int(np.ceil(idx))
                    alpha = idx - idx_low
                    resampled[i] = (1-alpha) * strokes[idx_low] + alpha * strokes[idx_high]
        
        return resampled
    
    def extract_features(self, strokes: np.ndarray) -> np.ndarray:
        """
        Extract rich features from stroke data.
        
        Args:
            strokes (np.ndarray): Resampled stroke data [x, y, stroke_id]
            
        Returns:
            np.ndarray: Feature matrix [x, y, vx, vy, speed, direction, curvature, pen_state]
        """
        n_points = len(strokes)
        
        # Ensure we have the exact target length
        if n_points != self.target_length:
            # This should not happen after proper resampling, but let's be safe
            strokes_fixed = np.zeros((self.target_length, 3))
            if n_points > 0:
                min_len = min(n_points, self.target_length)
                strokes_fixed[:min_len] = strokes[:min_len]
            strokes = strokes_fixed
            n_points = self.target_length
        
        features = np.zeros((n_points, self.feature_dim))
        
        # Basic coordinates
        features[:, 0] = strokes[:, 0]  # x
        features[:, 1] = strokes[:, 1]  # y
        
        # Velocity features
        if n_points > 1:
            # Forward differences for velocity
            dx = np.diff(strokes[:, 0])
            dy = np.diff(strokes[:, 1])
            
            # Ensure exact length match by padding properly
            dx_padded = np.zeros(n_points)
            dy_padded = np.zeros(n_points)
            
            # Place differences in appropriate positions
            dx_padded[:-1] = dx[:min(len(dx), n_points-1)]
            dy_padded[:-1] = dy[:min(len(dy), n_points-1)]
            
            # Extend last value for the final point
            if len(dx) > 0:
                dx_padded[-1] = dx_padded[-2] if n_points > 1 else 0
                dy_padded[-1] = dy_padded[-2] if n_points > 1 else 0
            
            features[:, 2] = dx_padded  # velocity_x
            features[:, 3] = dy_padded  # velocity_y
            
            # Speed (magnitude of velocity)
            features[:, 4] = np.sqrt(dx_padded**2 + dy_padded**2)
            
            # Direction (angle of velocity vector)
            features[:, 5] = np.arctan2(dy_padded, dx_padded)
        
        # Curvature (rate of direction change)
        if n_points > 2:
            directions = features[:, 5]
            # Handle angle wrapping
            direction_diff = np.diff(directions)
            direction_diff = np.mod(direction_diff + np.pi, 2*np.pi) - np.pi
            
            # Ensure exact length match
            curvature = np.zeros(n_points)
            curvature[1:-1] = direction_diff[:n_points-2]  # Place in middle
            curvature[-1] = curvature[-2] if n_points > 2 else 0  # Extend last value
            
            features[:, 6] = curvature
        
        # Pen state (stroke boundaries)
        # 1.0 = pen down, 0.0 = pen up (stroke transition)
        pen_state = np.ones(n_points)
        if n_points > 1:
            stroke_changes = np.diff(strokes[:, 2]) != 0
            pen_up_indices = np.where(stroke_changes)[0] + 1
            pen_state[pen_up_indices] = 0.0
        features[:, 7] = pen_state
        
        return features
    
    def augment_data(self, features: np.ndarray, 
                    rotation_range: float = 0.3, 
                    scale_range: float = 0.2,
                    noise_std: float = 0.05) -> np.ndarray:
        """
        Apply data augmentation to features.
        
        Args:
            features (np.ndarray): Original features
            rotation_range (float): Maximum rotation in radians
            scale_range (float): Maximum scale variation (±)
            noise_std (float): Standard deviation of Gaussian noise
            
        Returns:
            np.ndarray: Augmented features
        """
        augmented = features.copy()
        
        # Random rotation
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Rotate coordinates
            x_rot = augmented[:, 0] * cos_a - augmented[:, 1] * sin_a
            y_rot = augmented[:, 0] * sin_a + augmented[:, 1] * cos_a
            augmented[:, 0] = x_rot
            augmented[:, 1] = y_rot
            
            # Rotate velocity
            vx_rot = augmented[:, 2] * cos_a - augmented[:, 3] * sin_a
            vy_rot = augmented[:, 2] * sin_a + augmented[:, 3] * cos_a
            augmented[:, 2] = vx_rot
            augmented[:, 3] = vy_rot
            
            # Update direction
            augmented[:, 5] = np.arctan2(vy_rot, vx_rot)
        
        # Random scaling
        if scale_range > 0:
            scale = 1.0 + np.random.uniform(-scale_range, scale_range)
            augmented[:, :4] *= scale  # Scale coordinates and velocities
            augmented[:, 4] *= scale   # Scale speed
        
        # Add noise
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, augmented[:, :2].shape)
            augmented[:, :2] += noise
        
        return augmented
    
    def preprocess_stroke(self, file_path: Path, augment: bool = False) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Complete preprocessing pipeline for a single stroke file.
        
        Args:
            file_path (Path): Path to InkML file
            augment (bool): Whether to apply data augmentation
            
        Returns:
            Tuple[Optional[str], Optional[np.ndarray]]: (label, processed_features)
        """
        # Parse file
        label, strokes = self.parse_inkml_strokes(file_path)
        
        if label is None or strokes is None or len(strokes) == 0:
            return None, None
        
        # Apply preprocessing pipeline
        strokes = self.spatial_normalization(strokes)
        strokes = self.temporal_resampling(strokes)
        features = self.extract_features(strokes)
        
        # Optional augmentation
        if augment:
            features = self.augment_data(features)
        
        return label, features
    
    def get_data_statistics(self, features_list: List[np.ndarray]) -> Dict:
        """
        Compute dataset statistics for normalization.
        
        Args:
            features_list (List[np.ndarray]): List of feature arrays
            
        Returns:
            Dict: Statistics dictionary
        """
        if not features_list:
            return {}
        
        # Stack all features
        all_features = np.vstack(features_list)
        
        stats = {
            'mean': np.mean(all_features, axis=0),
            'std': np.std(all_features, axis=0) + 1e-8,  # Avoid division by zero
            'min': np.min(all_features, axis=0),
            'max': np.max(all_features, axis=0)
        }
        
        return stats
    
    def normalize_features(self, features: np.ndarray, stats: Dict) -> np.ndarray:
        """
        Normalize features using dataset statistics.
        
        Args:
            features (np.ndarray): Feature array
            stats (Dict): Statistics from get_data_statistics
            
        Returns:
            np.ndarray: Normalized features
        """
        if not stats:
            return features
        
        normalized = (features - stats['mean']) / stats['std']
        return normalized
    
    def preprocess_strokes(self, strokes_list: List) -> Optional[np.ndarray]:
        """
        Preprocess a list of strokes for model inference.
        
        Args:
            strokes_list (List): List of stroke arrays, each containing [x, y] coordinates
            
        Returns:
            Optional[np.ndarray]: Processed feature array of shape (target_length, feature_dim)
        """
        if not strokes_list:
            return None
        
        # Convert strokes list to numpy array format
        all_points = []
        stroke_id = 0
        
        for stroke in strokes_list:
            if len(stroke) == 0:
                continue
            stroke_array = np.array(stroke)
            if len(stroke_array.shape) == 2 and stroke_array.shape[1] >= 2:
                # Add stroke_id column
                points_with_id = np.column_stack([
                    stroke_array[:, 0],  # x
                    stroke_array[:, 1],  # y  
                    np.full(len(stroke_array), stroke_id)  # stroke_id
                ])
                all_points.append(points_with_id)
                stroke_id += 1
        
        if not all_points:
            return None
        
        # Combine all strokes
        strokes_array = np.vstack(all_points)
        
        # Apply preprocessing pipeline
        strokes_norm = self.spatial_normalization(strokes_array)
        strokes_resampled = self.temporal_resampling(strokes_norm)
        features = self.extract_features(strokes_resampled)
        
        return features
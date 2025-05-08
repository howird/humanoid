"""
Modified code from https://github.com/nwojke/deep_sort
"""

import copy
from collections import deque
from typing import Dict, List, Optional, Any, Union, Tuple, Deque

import numpy as np
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter1d


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A track class that represents a tracked object with history and prediction data.
    """

    def __init__(
        self,
        cfg,
        track_id: int,
        n_init: int,
        max_age: int,
        detection_data: Dict[str, Any],
        detection_id: Optional[int] = None,
        dims: Optional[List[int]] = None,
    ):
        self.cfg = cfg
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.time_init = detection_data["time"]
        self.state = TrackState.Tentative

        self._n_init = n_init
        self._max_age = max_age

        if dims is not None:
            self.A_dim = dims[0]
            self.P_dim = dims[1]
            self.L_dim = dims[2]

        self.track_data: Dict[str, Any] = {"history": deque(maxlen=self.cfg.phalp.track_history), "prediction": {}}
        for _ in range(self.cfg.phalp.track_history):
            self.track_data["history"].append(detection_data)

        self.track_data["prediction"]["appe"] = deque([detection_data["appe"]], maxlen=self.cfg.phalp.n_init + 1)
        self.track_data["prediction"]["loca"] = deque([detection_data["loca"]], maxlen=self.cfg.phalp.n_init + 1)
        self.track_data["prediction"]["pose"] = deque([detection_data["pose"]], maxlen=self.cfg.phalp.n_init + 1)
        self.track_data["prediction"]["uv"] = deque(
            [copy.deepcopy(detection_data["uv"])], maxlen=self.cfg.phalp.n_init + 1
        )

        # if the track is initialized by detection with annotation, then we set the track state to confirmed
        if len(detection_data["annotations"]) > 0:
            self.state = TrackState.Confirmed

    def update_track(
        self, 
        detection: Optional[Any] = None, 
        detection_id: Optional[int] = None, 
        shot: Optional[int] = None,
        predicted_features: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """
        Update the track state. This function combines the previous predict and update steps.
        
        Args:
            detection: The detection to update with (None if just predicting)
            detection_id: ID of the detection
            shot: Shot change indicator (1 if shot change, 0 otherwise)
            predicted_features: Dictionary of predicted features (appe, pose, loca, uv)
                                If None, features will be copied from the latest history
        """
        # Always increment age
        self.age += 1
        
        # If we have a detection, update with it
        if detection is not None:
            # Update history with new detection
            self.track_data["history"].append(copy.deepcopy(detection.detection_data))
            
            # Handle shot changes
            if shot == 1:
                for tx in range(self.cfg.phalp.track_history):
                    self.track_data["history"][-1 - tx]["loca"] = copy.deepcopy(detection.detection_data["loca"])

            # Update UV prediction
            if "T" in self.cfg.phalp.predict:
                mixing_alpha_ = self.cfg.phalp.alpha * (detection.detection_data["conf"] ** 2)
                ones_old = self.track_data["prediction"]["uv"][-1][3:, :, :] == 1
                ones_new = self.track_data["history"][-1]["uv"][3:, :, :] == 1
                ones_old = np.repeat(ones_old, 3, 0)
                ones_new = np.repeat(ones_new, 3, 0)
                ones_intersect = np.logical_and(ones_old, ones_new)
                ones_union = np.logical_or(ones_old, ones_new)
                good_old_ones = np.logical_and(np.logical_not(ones_intersect), ones_old)
                good_new_ones = np.logical_and(np.logical_not(ones_intersect), ones_new)
                new_rgb_map = np.zeros((3, 256, 256))
                new_mask_map = np.zeros((1, 256, 256)) - 1
                new_mask_map[ones_union[:1, :, :]] = 1.0
                new_rgb_map[ones_intersect] = (1 - mixing_alpha_) * self.track_data["prediction"]["uv"][-1][:3, :, :][
                    ones_intersect
                ] + mixing_alpha_ * self.track_data["history"][-1]["uv"][:3, :, :][ones_intersect]
                new_rgb_map[good_old_ones] = self.track_data["prediction"]["uv"][-1][:3, :, :][good_old_ones]
                new_rgb_map[good_new_ones] = self.track_data["history"][-1]["uv"][:3, :, :][good_new_ones]
                self.track_data["prediction"]["uv"].append(np.concatenate((new_rgb_map, new_mask_map), 0))
            else:
                self.track_data["prediction"]["uv"].append(self.track_data["history"][-1]["uv"])

            # Update track state
            self.hits += 1
            self.time_since_update = 0
            
            # Confirm track if needed
            if self.state == TrackState.Tentative and self.hits >= self._n_init:
                self.state = TrackState.Confirmed

            # If the detection has annotation, set the track state to confirmed
            if len(detection.detection_data["annotations"]) > 0:
                self.state = TrackState.Confirmed
        else:
            # No detection, just prediction
            self.time_since_update += 1
        
        # Update predictions if provided
        if predicted_features is not None:
            self._update_predictions(predicted_features)
    
    def _update_predictions(self, predicted_features: Dict[str, Optional[np.ndarray]]) -> None:
        """
        Update the track's predictions with new feature predictions.
        
        Args:
            predicted_features: Dictionary with keys 'appe', 'pose', 'loca', 'uv' and their predicted values
        """
        # Get the latest history features as fallback
        latest_features = self.track_data["history"][-1]
        
        # Update each feature type
        for feature_type in ['appe', 'pose', 'loca']:
            if feature_type in predicted_features and predicted_features[feature_type] is not None:
                # Use the provided prediction
                feature_predicted = copy.deepcopy(predicted_features[feature_type].numpy())
            else:
                # Fall back to the latest history
                feature_predicted = copy.deepcopy(latest_features[feature_type])
                
            self.track_data["prediction"][feature_type].append(feature_predicted)
        
        # Handle UV separately if provided
        if 'uv' in predicted_features and predicted_features['uv'] is not None:
            self.track_data["prediction"]["uv"].append(predicted_features['uv'])
            
    def add_predicted(self, appe: Optional[np.ndarray] = None, 
                     pose: Optional[np.ndarray] = None, 
                     loca: Optional[np.ndarray] = None, 
                     uv: Optional[np.ndarray] = None) -> None:
        """
        Add predicted features to the track.
        
        Args:
            appe: Appearance embedding
            pose: Pose embedding
            loca: Location embedding
            uv: UV map
        """
        # Create a dictionary of predictions
        predicted_features = {
            "appe": appe,
            "pose": pose,
            "loca": loca,
            "uv": uv
        }
        
        # Update predictions using the existing method
        self._update_predictions(predicted_features)

    def mark_missed(self) -> None:
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self) -> bool:
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> bool:
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def smooth_bbox(self, bbox: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply smoothing to bounding box coordinates.

        Args:
            bbox: List of bounding box coordinates

        Returns:
            Smoothed bounding box coordinates
        """
        kernel_size = 5
        sigma = 3
        bbox = np.array(bbox)
        smoothed = np.array([signal.medfilt(param, kernel_size) for param in bbox.T]).T
        out = np.array([gaussian_filter1d(traj, sigma) for traj in smoothed.T]).T
        return list(out)

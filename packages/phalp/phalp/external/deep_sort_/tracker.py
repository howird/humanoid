"""
Modified code from https://github.com/nwojke/deep_sort
"""

from __future__ import absolute_import
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union, Set, NamedTuple
import numpy as np
import torch

from . import linear_assignment
from .track import Track

np.set_printoptions(formatter={"float": "{: 0.3f}".format})


@dataclass
class TrackerDimensions:
    """Dimensions for the tracker features"""

    appearance_dim: int
    pose_dim: int
    location_dim: int


@dataclass
class TrackingStatistics:
    """Statistics about the tracking process"""

    cost_matrix: np.ndarray
    track_ground_truths: List[Any]
    detection_ground_truths: List[Any]
    track_indices: List[int]
    detection_indices: List[int]


@dataclass
class TrackingResult:
    """Result of a tracking update"""

    tracks: List[Track]
    matches: List[Tuple[int, int]]
    unmatched_tracks: List[int]
    unmatched_detections: List[int]
    statistics: TrackingStatistics
    active_tracks: List[Track]


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    def __init__(self, cfg, metric, max_age=30, n_init=3, phalp_tracker=None, dims=None):
        self.cfg = cfg
        self.metric = metric
        self.max_age = max_age
        self.n_init = n_init
        self.tracks: List[Track] = []
        self._next_id = 1
        self.tracked_cost: Dict[int, List] = {}
        self.phalp_tracker = phalp_tracker

        if dims is not None:
            self.dimensions = TrackerDimensions(appearance_dim=dims[0], pose_dim=dims[1], location_dim=dims[2])

    def process_frame(self, detections, frame_t, image_name, shot) -> TrackingResult:
        """
        Process a new frame - predict track states and update with detections.
        This combines the previous predict and update steps into a single function.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        frame_t : int
            Current frame number
        image_name : str
            Name of the current image
        shot : int
            Shot change indicator

        Returns
        -------
        TrackingResult
            Result of the tracking update containing tracks, matches, and statistics
        """
        # First, match detections to existing tracks
        matches, unmatched_tracks, unmatched_detections, statistics = self._match(detections)

        # Store tracking statistics
        self.tracked_cost[frame_t] = [
            statistics.cost_matrix,
            matches,
            unmatched_tracks,
            unmatched_detections,
            statistics.track_ground_truths,
            statistics.detection_ground_truths,
            statistics.track_indices,
            statistics.detection_indices,
        ]

        if self.cfg.verbose:
            print(np.round(np.array(statistics.cost_matrix), 2))

        # Get predicted features for all tracks that need updating
        all_track_indices = [idx for idx, _ in matches] + unmatched_tracks
        all_predicted_features = self._predict_track_features(all_track_indices, self.cfg.phalp.predict)

        # Create a dictionary mapping track indices to their predicted features
        track_predictions = {}
        for i, track_idx in enumerate(all_track_indices):
            if i < len(all_predicted_features):
                track_predictions[track_idx] = all_predicted_features[i]

        # Update matched tracks with their detections
        for track_idx, detection_idx in matches:
            predicted_features_dict = track_predictions.get(track_idx)

            # Update the track with both detection and predictions
            self.tracks[track_idx].update_track(
                detection=detections[detection_idx],
                detection_id=detection_idx,
                shot=shot,
                predicted_features=predicted_features_dict,
            )

        # Update unmatched tracks (prediction only)
        for track_idx in unmatched_tracks:
            predicted_features_dict = track_predictions.get(track_idx)

            # Update the track with predictions only (no detection)
            self.tracks[track_idx].update_track(detection=None, predicted_features=predicted_features_dict)

            # Mark as missed
            self.tracks[track_idx].mark_missed()

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], detection_idx)

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Get active tracks (confirmed or tentative)
        active_tracks = [t for t in self.tracks if t.is_confirmed() or t.is_tentative()]
        active_targets = [t.track_id for t in active_tracks]

        # Update metric with features from active tracks
        self._update_metric(active_tracks, active_targets)

        return TrackingResult(
            tracks=self.tracks,
            matches=matches,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_detections,
            statistics=statistics,
            active_tracks=active_tracks,
        )

    def _update_metric(self, active_tracks: List[Track], active_targets: List[int]) -> None:
        """Update the distance metric with features from active tracks.

        Parameters
        ----------
        active_tracks : List[Track]
            List of active tracks
        active_targets : List[int]
            List of active track IDs
        """
        appe_features, loca_features, pose_features, uv_maps, targets = [], [], [], [], []

        for track in active_tracks:
            appe_features.append(track.track_data["prediction"]["appe"][-1])
            loca_features.append(track.track_data["prediction"]["loca"][-1])
            pose_features.append(track.track_data["prediction"]["pose"][-1])
            uv_maps.append(track.track_data["prediction"]["uv"][-1])
            targets.append(track.track_id)

        self.metric.partial_fit(
            np.asarray(appe_features),
            np.asarray(loca_features),
            np.asarray(pose_features),
            np.asarray(uv_maps),
            np.asarray(targets),
            active_targets,
        )

    def _match(self, detections) -> Tuple[List[Tuple[int, int]], List[int], List[int], TrackingStatistics]:
        """Match detections to tracks.

        Parameters
        ----------
        detections : List[Detection]
            List of detections

        Returns
        -------
        Tuple[List[Tuple[int, int]], List[int], List[int], TrackingStatistics]
            Matches, unmatched tracks, unmatched detections, and statistics
        """

        def gated_metric(tracks, dets, track_indices, detection_indices):
            appe_emb = np.array([dets[i].detection_data["appe"] for i in detection_indices])
            loca_emb = np.array([dets[i].detection_data["loca"] for i in detection_indices])
            pose_emb = np.array([dets[i].detection_data["pose"] for i in detection_indices])
            uv_maps = np.array([dets[i].detection_data["uv"] for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(
                [appe_emb, loca_emb, pose_emb, uv_maps],
                targets,
                dims=[self.dimensions.appearance_dim, self.dimensions.pose_dim, self.dimensions.location_dim],
                phalp_tracker=self.phalp_tracker,
            )

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]

        # Associate confirmed tracks using appearance features.
        matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(
            gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks
        )

        track_gt = [
            t.track_data["history"][-1]["ground_truth"]
            for i, t in enumerate(self.tracks)
            if t.is_confirmed() or t.is_tentative()
        ]
        detect_gt = [d.detection_data["ground_truth"] for i, d in enumerate(detections)]

        track_idt = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_idt = [i for i, d in enumerate(detections)]

        statistics = TrackingStatistics(
            cost_matrix=cost_matrix,
            track_ground_truths=track_gt,
            detection_ground_truths=detect_gt,
            track_indices=track_idt,
            detection_indices=detect_idt,
        )

        if self.cfg.use_gt:
            matches = []
            for t_, t_gt in enumerate(track_gt):
                for d_, d_gt in enumerate(detect_gt):
                    if t_gt == d_gt:
                        matches.append([t_, d_])
            t_pool = [t_ for (t_, _) in matches]
            d_pool = [d_ for (_, d_) in matches]
            unmatched_tracks = [t_ for t_ in track_idt if t_ not in t_pool]
            unmatched_detections = [d_ for d_ in detect_idt if d_ not in d_pool]

        return matches, unmatched_tracks, unmatched_detections, statistics

    def _initiate_track(self, detection, detection_id) -> Track:
        """Create a new track from a detection.

        Parameters
        ----------
        detection : Detection
            Detection to create a track from
        detection_id : int
            ID of the detection

        Returns
        -------
        Track
            Newly created track
        """
        new_track = Track(
            self.cfg,
            self._next_id,
            self.n_init,
            self.max_age,
            detection_data=detection.detection_data,
            detection_id=detection_id,
            dims=[self.dimensions.appearance_dim, self.dimensions.pose_dim, self.dimensions.location_dim],
        )
        new_track.add_predicted()
        self.tracks.append(new_track)
        self._next_id += 1
        return new_track

    def _predict_track_features(self, track_ids, features="APL"):
        """
        Predict features for the specified tracks using the PHALP tracker.

        This method extracts historical data from tracks and passes it to the PHALP
        tracker's forward_for_tracking method to predict future states.

        Parameters
        ----------
        track_ids : List[int]
            List of track indices
        features : str
            String indicating which features to predict:
            - "A": Appearance features (currently not predicted, just copied)
            - "P": Pose features (predicted using pose transformer)
            - "L": Location features (predicted using linear regression)

        Returns
        -------
        List[Dict[str, np.ndarray]]
            List of dictionaries containing predicted features for each track
        """
        # If no tracks to predict, return empty list
        if not track_ids:
            return []

        # Extract track history data
        track_history_data = self._extract_track_history_data(track_ids, features)

        # If no valid tracks, return empty list
        if not track_history_data["is_tracks"]:
            return []

        # Get predictions from PHALP tracker
        predictions = self._get_predictions_from_phalp(track_ids, features, track_history_data)

        return predictions

    def _extract_track_history_data(self, track_ids, features):
        """
        Extract historical data from tracks for prediction.

        Parameters
        ----------
        track_ids : List[int]
            List of track indices
        features : str
            String indicating which features to extract

        Returns
        -------
        Dict
            Dictionary containing extracted historical data
        """
        a_features = []
        p_features = []
        l_features = []
        t_features = []
        l_time = []
        confidence = []
        is_tracks = 0
        p_data = []

        for track_idx in track_ids:
            t_features.append(
                [self.tracks[track_idx].track_data["history"][i]["time"] for i in range(self.cfg.phalp.track_history)]
            )
            l_time.append(self.tracks[track_idx].time_since_update)

            if "L" in features:
                l_features.append(
                    np.array(
                        [
                            self.tracks[track_idx].track_data["history"][i]["loca"]
                            for i in range(self.cfg.phalp.track_history)
                        ]
                    )
                )
            if "P" in features:
                p_features.append(
                    np.array(
                        [
                            self.tracks[track_idx].track_data["history"][i]["pose"]
                            for i in range(self.cfg.phalp.track_history)
                        ]
                    )
                )
            if "P" in features:
                t_id = self.tracks[track_idx].track_id
                p_data.append(
                    [
                        [data["xy"][0], data["xy"][1], data["scale"], data["scale"], data["time"], t_id]
                        for data in self.tracks[track_idx].track_data["history"]
                    ]
                )
            if "L" in features:
                confidence.append(
                    np.array(
                        [
                            self.tracks[track_idx].track_data["history"][i]["conf"]
                            for i in range(self.cfg.phalp.track_history)
                        ]
                    )
                )
            is_tracks = 1

        # Convert to numpy arrays
        l_time = np.array(l_time)
        t_features = np.array(t_features)
        if "P" in features:
            p_features = np.array(p_features)
        if "P" in features:
            p_data = np.array(p_data)
        if "L" in features:
            l_features = np.array(l_features)
        if "L" in features:
            confidence = np.array(confidence)

        return {
            "p_features": p_features if "P" in features else None,
            "p_data": p_data if "P" in features else None,
            "l_features": l_features if "L" in features else None,
            "t_features": t_features,
            "l_time": l_time,
            "confidence": confidence if "L" in features else None,
            "is_tracks": is_tracks,
        }

    def _get_predictions_from_phalp(self, track_ids, features, track_history_data):
        """
        Get predictions from PHALP tracker using track history data.

        Parameters
        ----------
        track_ids : List[int]
            List of track indices
        features : str
            String indicating which features to predict
        track_history_data : Dict
            Dictionary containing track history data

        Returns
        -------
        List[Dict[str, np.ndarray]]
            List of dictionaries containing predicted features for each track
        """
        predictions = []

        with torch.no_grad():
            p_pred = None
            l_pred = None

            if "P" in features:
                p_pred = self.phalp_tracker.forward_for_tracking(
                    [track_history_data["p_features"], track_history_data["p_data"], track_history_data["t_features"]],
                    "P",
                    track_history_data["l_time"],
                )
            if "L" in features:
                l_pred = self.phalp_tracker.forward_for_tracking(
                    [
                        track_history_data["l_features"],
                        track_history_data["t_features"],
                        track_history_data["confidence"],
                    ],
                    "L",
                    track_history_data["l_time"],
                )

        for p_id, track_idx in enumerate(track_ids):
            # Create a dictionary of predictions for this track
            track_predictions = {
                "pose": p_pred[p_id] if ("P" in features and p_pred is not None) else None,
                "loca": l_pred[p_id] if ("L" in features and l_pred is not None) else None,
                "appe": None,  # We don't predict appearance currently
                "uv": None,  # UV is handled separately
            }

            # Add to the predictions list
            predictions.append(track_predictions)

        return predictions

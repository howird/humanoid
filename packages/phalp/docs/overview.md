# PHALP: People Detection and Tracking Pipeline

## Overview

PHALP (Persistent Human Appearance and Location Prediction) is a framework for detecting, tracking, and predicting human appearance and motion in videos. The pipeline integrates multiple computer vision components including detection, pose estimation, appearance modeling, and tracking.

## Pipeline Architecture

The main tracking pipeline is implemented in the `track()` method of the `PHALP` class. Below is a detailed breakdown of each stage:

### 1. Initialization and Setup

- **Input**: Configuration object (`BaseConfig`)
- **Output**: Initialized models and trackers
- **Process**: 
  - Downloads required model weights
  - Initializes HMR (Human Mesh Recovery) model
  - Sets up pose predictor for temporal prediction
  - Initializes object detector (Detectron2)
  - Configures DeepSort tracker

### 2. Video Processing

- **Input**: Video file, YouTube link, or folder of images
- **Output**: List of frames to process
- **Process**: The `IOManager` handles reading frames from various sources

### 3. Detection Stage

- **Input**: Single video frame (shape: [H, W, 3])
- **Output**: 
  - Bounding boxes (shape: [N, 4]) where N is number of detections
  - Segmentation masks (shape: [N, H, W])
  - Confidence scores (shape: [N])
  - Class IDs (shape: [N])
- **Process**: Uses Detectron2 to detect people in the frame

### 4. Human Feature Extraction

- **Input**: 
  - Image frame
  - Detected bounding boxes and masks
- **Output**: 
  - List of `Detection` objects containing:
    - Appearance embeddings (shape: [4096])
    - Pose embeddings (shape: [4096] or joint-specific)
    - Location embeddings (shape: [99])
    - UV maps (texture maps)
    - SMPL parameters
- **Process**: 
  - Crops detected people
  - Processes through HMAR model
  - Extracts appearance, pose, and location features

### 5. Tracking

- **Input**: List of `Detection` objects
- **Output**: 
  - Matches between tracks and detections
  - Updated track states
- **Process**:
  - Predicts new locations of existing tracks
  - Matches detections to existing tracks using appearance, pose, and location features
  - Updates matched tracks with new observations
  - Creates new tracks for unmatched detections
  - Predicts future states for tracks with missing detections

### 6. Visualization and Storage

- **Input**: Tracking results
- **Output**: 
  - Rendered video
  - Pickle file with tracking data
- **Process**:
  - Renders tracked people with 3D meshes
  - Stores tracking data in a structured format

## Output Data Format

The tracking results are stored in a pickle file with the following structure:

```
{
    frame_path: {
        "time": frame_index,
        "shot": shot_change_flag,
        "frame_path": path_to_frame,
        "frame": image_data (if rendering enabled),
        "tid": list of track IDs,
        "bbox": list of bounding boxes,
        "tracked_time": list of time since last update,
        "tracked_ids": list of confirmed track IDs,
        "tracked_bbox": list of confirmed bounding boxes,
        
        # Feature data
        "appe": list of appearance embeddings,
        "loca": list of location embeddings,
        "pose": list of pose embeddings,
        "uv": list of UV maps,
        
        # Prediction data (if enabled)
        "prediction_uv": list of predicted UV maps,
        "prediction_pose": list of predicted poses,
        "prediction_loca": list of predicted locations,
        
        # Additional data
        "center": list of detection centers,
        "scale": list of detection scales,
        "size": list of image sizes,
        "img_path": list of image paths,
        "img_name": list of image names,
        "class_name": list of class names,
        "conf": list of confidence scores,
        "annotations": list of annotations,
        "smpl": list of SMPL parameters,
        "camera": list of camera parameters,
        "camera_bbox": list of camera parameters in bbox space,
        "3d_joints": list of 3D joint positions,
        "2d_joints": list of 2D joint positions,
        "mask": list of segmentation masks,
        "extra_data": list of additional data
    }
}
```

The keys are selected based on the tracking configuration and can be customized by modifying the `visual_store_` list in the `track()` method.

from __future__ import absolute_import, division, print_function

import copy
import os
import pickle

from pathlib import Path
from typing import List

import numpy as np
import scipy.stats as stats
import torch
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from phalp.configs.base import CACHE_DIR
from phalp.utils.utils_download import cache_url
from phalp.utils.colors import phalp_colors, slahmr_colors
from phalp.utils import get_pylogger

log = get_pylogger(__name__)


def get_progress_bar(sequence, total=None, description=None, disable=False):
    columns: List["ProgressColumn"] = [TextColumn("[progress.description]{task.description}")] if description else []
    columns.extend(
        (
            SpinnerColumn(spinner_name="runner"),
            BarColumn(
                style="bar.back",
                complete_style="bar.complete",
                finished_style="bar.finished",
                pulse_style="bar.pulse",
            ),
            TaskProgressColumn(show_speed=True),
            "eta :",
            TimeRemainingColumn(),  # elapsed_when_finished=True
            " time elapsed :",
            TimeElapsedColumn(),
        )
    )

    progress_bar = Progress(
        *columns,
        auto_refresh=True,
        console=None,
        transient=False,
        get_time=None,
        refresh_per_second=10.0,
        disable=disable,
    )

    return progress_bar


def progress_bar(sequence, total=None, description=None, disable=False):
    progress_bar = get_progress_bar(sequence, total, description, disable)
    with progress_bar:
        yield from progress_bar.track(sequence, total=total, description=description, update_period=0.1)


def numpy_to_torch_image(ndarray):
    torch_image = torch.from_numpy(ndarray)
    torch_image = torch_image.unsqueeze(0)
    torch_image = torch_image.permute(0, 3, 1, 2)
    torch_image = torch_image[:, [2, 1, 0], :, :]
    return torch_image


def get_colors(pallette="phalp"):
    try:
        if pallette == "phalp":
            colors = phalp_colors
        elif pallette == "slahmr":
            colors = slahmr_colors
        else:
            raise ValueError("Invalid pallette")

        RGB_tuples = np.vstack([colors, np.random.uniform(0, 255, size=(10000, 3)), [[0, 0, 0]]])
        b = np.where(RGB_tuples == 0)
        RGB_tuples[b] = 1
    except:
        from colordict import ColorDict

        colormap = np.array(
            list(ColorDict(norm=255, mode="rgb", palettes_path="", is_grayscale=False, palettes="all").values())
        )
        RGB_tuples = np.vstack([colormap[1:, :3], np.random.uniform(0, 255, size=(10000, 3)), [[0, 0, 0]]])

    return RGB_tuples


def task_divider(data, batch_id, num_task):
    batch_length = len(data) // num_task
    start_ = batch_id * (batch_length + 1)
    end_ = (batch_id + 1) * (batch_length + 1)
    if start_ > len(data):
        exit()
    if end_ > len(data):
        end_ = len(data)
    data = data[start_:end_] if batch_id >= 0 else data

    return data


def get_prediction_interval(y, y_hat, x, x_hat):
    n = y.size
    resid = y - y_hat
    s_err = np.sqrt(np.sum(resid**2) / (n - 2))  # standard deviation of the error
    t = stats.t.ppf(0.975, n - 2)  # used for CI and PI bands
    pi = t * s_err * np.sqrt(1 + 1 / n + (x_hat - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    return pi


def pose_camera_vector_to_smpl(pose_camera_vector):
    # pose_camera_vector: [229,]
    global_orient = pose_camera_vector[:9].reshape(1, 3, 3)
    body_pose = pose_camera_vector[9 : 9 + 207].reshape(23, 3, 3)
    betas = pose_camera_vector[9 + 207 : 9 + 207 + 10].reshape(
        10,
    )
    camera = pose_camera_vector[9 + 207 + 10 : 9 + 207 + 10 + 3].reshape(1, 3)
    camera[:, 2] *= 200.0
    return {"global_orient": global_orient, "body_pose": body_pose, "betas": betas}, camera[0]


def smpl_to_pose_camera_vector(smpl_params, camera):
    # convert smpl parameters to camera to pose_camera_vector for smoothness.
    global_orient_ = smpl_params["global_orient"].reshape(1, -1)  # 1x3x3 -> 9
    body_pose_ = smpl_params["body_pose"].reshape(1, -1)  # 23x3x3 -> 207
    shape_ = smpl_params["betas"].reshape(1, -1)  # 10 -> 10
    loca_ = copy.deepcopy(camera.view(1, -1))  # 3 -> 3
    loca_[:, 2] = loca_[:, 2] / 200.0
    pose_embedding = np.concatenate((global_orient_, body_pose_, shape_, loca_.cpu().numpy()), 1)
    return pose_embedding


def convert_pkl(old_pkl):
    # Code adapted from https://github.com/nkolot/ProHMR
    # Convert SMPL pkl file to be compatible with Python 3
    # Script is from https://rebeccabilbro.github.io/convert-py2-pickles-to-py3/

    # Make a name for the new pickle
    new_pkl = os.path.splitext(os.path.basename(old_pkl))[0] + "_p3.pkl"

    # Convert Python 2 "ObjectType" to Python 3 object
    dill._dill._reverse_typemap["ObjectType"] = object

    # Open the pickle using latin1 encoding
    with open(old_pkl, "rb") as f:
        loaded = pickle.load(f, encoding="latin1")

    # Re-save as Python 3 pickle
    with open(new_pkl, "wb") as outfile:
        pickle.dump(loaded, outfile)


def perspective_projection(points, rotation, translation, focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:, 0, 0] = focal_length[:, 0]
    K[:, 1, 1] = focal_length[:, 1]
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum("bij,bkj->bki", rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / (points[:, :, -1].unsqueeze(-1))

    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)

    return projected_points[:, :, :-1]


def compute_uvsampler(vt, ft, tex_size=6):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    uv = obj2nmr_uvmap(ft, vt, tex_size=tex_size)
    uv = uv.reshape(-1, tex_size, tex_size, 2)
    return uv


def obj2nmr_uvmap(ft, vt, tex_size=6):
    """
    Converts obj uv_map to NMR uv_map (F x T x T x 2),
    where tex_size (T) is the sample rate on each face.
    """
    # This is F x 3 x 2
    uv_map_for_verts = vt[ft]

    # obj's y coordinate is [1-0], but image is [0-1]
    uv_map_for_verts[:, :, 1] = 1 - uv_map_for_verts[:, :, 1]

    # range [0, 1] -> [-1, 1]
    uv_map_for_verts = (2 * uv_map_for_verts) - 1

    alpha = np.arange(tex_size, dtype=np.float64) / (tex_size - 1)
    beta = np.arange(tex_size, dtype=np.float64) / (tex_size - 1)
    import itertools

    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])

    # Compute alpha, beta (this is the same order as NMR)
    v2 = uv_map_for_verts[:, 2]
    v0v2 = uv_map_for_verts[:, 0] - uv_map_for_verts[:, 2]
    v1v2 = uv_map_for_verts[:, 1] - uv_map_for_verts[:, 2]
    # Interpolate the vertex uv values: F x 2 x T*2
    uv_map = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 2, 1)

    # F x T*2 x 2  -> F x T x T x 2
    uv_map = np.transpose(uv_map, (0, 2, 1)).reshape(-1, tex_size, tex_size, 2)

    return uv_map


def cached_download_from_drive(additional_urls=None):
    """Download a file from Google Drive if it doesn't exist yet.
    :param url: the URL of the file to download
    :param path: the path to save the file to
    """
    # Create necessary directories
    (CACHE_DIR / "phalp").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "phalp/3D").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "phalp/weights").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "phalp/ava").mkdir(parents=True, exist_ok=True)

    additional_urls = additional_urls if additional_urls is not None else {}
    download_files = {
        "head_faces.npy": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/head_faces.npy",
            str(CACHE_DIR / "phalp/3D"),
        ],
        "mean_std.npy": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/mean_std.npy",
            str(CACHE_DIR / "phalp/3D"),
        ],
        "smpl_mean_params.npz": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/smpl_mean_params.npz",
            str(CACHE_DIR / "phalp/3D"),
        ],
        "SMPL_to_J19.pkl": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/SMPL_to_J19.pkl",
            str(CACHE_DIR / "phalp/3D"),
        ],
        "texture.npz": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/texture.npz",
            str(CACHE_DIR / "phalp/3D"),
        ],
        "bmap_256.npy": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/bmap_256.npy",
            str(CACHE_DIR / "phalp/3D"),
        ],
        "fmap_256.npy": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/fmap_256.npy",
            str(CACHE_DIR / "phalp/3D"),
        ],
        "hmar_v2_weights.pth": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/hmar_v2_weights.pth",
            str(CACHE_DIR / "phalp/weights"),
        ],
        "pose_predictor.pth": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/pose_predictor_40006.ckpt",
            str(CACHE_DIR / "phalp/weights"),
        ],
        "pose_predictor.yaml": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/config_40006.yaml",
            str(CACHE_DIR / "phalp/weights"),
        ],
        # data for ava dataset
        "ava_labels.pkl": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_labels.pkl",
            str(CACHE_DIR / "phalp/ava"),
        ],
        "ava_class_mapping.pkl": [
            "https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_class_mappping.pkl",
            str(CACHE_DIR / "phalp/ava"),
        ],
    }
    download_files.update(additional_urls)

    for file_name, url in download_files.items():
        file_path = Path(url[1]) / file_name
        if not file_path.exists():
            print(f"Downloading file: {file_name}")
            # output = gdown.cached_download(url[0], str(file_path), fuzzy=True)
            output = cache_url(url[0], str(file_path))
            assert Path(output).exists(), f"{output} does not exist"

from typing import Optional
import numpy as np
import torch

from sklearn.linear_model import Ridge

from humanoid_vision.models.predictors.pose_transformer_v2 import PoseTransformerV2
from humanoid_vision.utils.utils import get_prediction_interval


def predict_future_pose(p_features, p_data, t_feature, time, pose_predictor: PoseTransformerV2):
    en_pose = torch.from_numpy(p_features)
    en_data = torch.from_numpy(p_data)
    en_time = torch.from_numpy(t_feature)

    if len(en_pose.shape) != 3:
        en_pose = en_pose.unsqueeze(0)  # (num_valid_persons, 7, pose_dim)
        en_time = en_time.unsqueeze(0)  # (num_valid_persons, 7)
        en_data = en_data.unsqueeze(0)  # (num_valid_persons, 7, 6)

    with torch.no_grad():
        pose_pred = pose_predictor.predict_next(en_pose, en_data, en_time, time)

    return pose_pred.cpu()


def predict_future_location(l_features, t_feature, confidence, time, distance_type):
    en_loca = torch.from_numpy(l_features)
    en_time = torch.from_numpy(t_feature)
    en_conf = torch.from_numpy(confidence)
    time = torch.from_numpy(time)

    if len(en_loca.shape) != 3:
        en_loca = en_loca.unsqueeze(0)
        en_time = en_time.unsqueeze(0)
    else:
        en_loca = en_loca.permute(0, 1, 2)

    num_valid_persons = en_loca.size(0)
    t = en_loca.size(1)

    en_loca_xy = en_loca[:, :, :90]
    en_loca_xy = en_loca_xy.view(num_valid_persons, t, 45, 2)
    en_loca_n = en_loca[:, :, 90:]
    en_loca_n = en_loca_n.view(num_valid_persons, t, 3, 3)

    new_en_loca_n = []
    for bs in range(num_valid_persons):
        x0_ = np.array(en_loca_xy[bs, :, 44, 0])
        y0_ = np.array(en_loca_xy[bs, :, 44, 1])
        n_ = np.log(np.array(en_loca_n[bs, :, 0, 2]))
        t = np.array(en_time[bs, :])

        loc_ = torch.diff(en_time[bs, :], dim=0) != 0
        if distance_type == "EQ_020" or distance_type == "EQ_021":
            loc_ = 1
        else:
            loc_ = loc_.shape[0] - torch.sum(loc_) + 1

        M = t[:, np.newaxis] ** [0, 1]
        time_ = 48 if time[bs] > 48 else time[bs]

        clf = Ridge(alpha=5.0)
        clf.fit(M, n_)
        n_p = clf.predict(np.array([1, time_ + 1 + t[-1]]).reshape(1, -1))
        n_p = n_p[0]
        n_hat = clf.predict(np.hstack((np.ones((t.size, 1)), t.reshape((-1, 1)))))
        n_pi = get_prediction_interval(n_, n_hat, t, time_ + 1 + t[-1])

        clf = Ridge(alpha=1.2)
        clf.fit(M, x0_)
        x_p = clf.predict(np.array([1, time_ + 1 + t[-1]]).reshape(1, -1))
        x_p = x_p[0]
        x_p_ = (x_p - 0.5) * np.exp(n_p) / 5000.0 * 256.0
        x_hat = clf.predict(np.hstack((np.ones((t.size, 1)), t.reshape((-1, 1)))))
        x_pi = get_prediction_interval(x0_, x_hat, t, time_ + 1 + t[-1])

        clf = Ridge(alpha=2.0)
        clf.fit(M, y0_)
        y_p = clf.predict(np.array([1, time_ + 1 + t[-1]]).reshape(1, -1))
        y_p = y_p[0]
        y_p_ = (y_p - 0.5) * np.exp(n_p) / 5000.0 * 256.0
        y_hat = clf.predict(np.hstack((np.ones((t.size, 1)), t.reshape((-1, 1)))))
        y_pi = get_prediction_interval(y0_, y_hat, t, time_ + 1 + t[-1])

        new_en_loca_n.append([x_p_, y_p_, np.exp(n_p), x_pi / loc_, y_pi / loc_, np.exp(n_pi) / loc_, 1, 1, 0])
        en_loca_xy[bs, -1, 44, 0] = x_p
        en_loca_xy[bs, -1, 44, 1] = y_p

    new_en_loca_n = torch.from_numpy(np.array(new_en_loca_n))
    xt = torch.cat(
        (
            en_loca_xy[:, -1, :, :].view(num_valid_persons, 90),
            (new_en_loca_n.float()).view(num_valid_persons, 9),
        ),
        1,
    )

    return xt

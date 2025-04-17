# humanoid

experimenting with controllers for physics-simulated humanoids

## Getting Started

1. Clone this repository.

```
git clone https://github.com/howird/humanoid.git
```

2. Install
   [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods)

3. Download and unzip Isaac Gym from
   [here](https://developer.nvidia.com/isaac-gym) and copy the contents of the
   python package to the isaacgym workspace folder:

```
cp -r <PATH TO EXTRACTED ISAACGYM>/isaacgym/python/* packages/isaacgym/
```

4. Create, activate, and sync your uv virtualenv

```
uv venv
source .venv/bin/activate
uv sync
```

5. Download the SMPL parameters from [SMPL](https://smpl.is.tue.mpg.de/), and
   unzip them into `data/smpl` folder. Rename the files
   `basicmodel_neutral_lbs_10_207_0_v1.1.0`,
   `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`,
   `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl`
   and `SMPL_FEMALE.pkl`.

6. Train a policy. In the virtual environment, run:

```
python scripts/train.py --env.motion_file <MOTION FILE PATH>
```

    The script supports wandb logging. To use wandb, log in to wandb, then add `--track` to the command.

    To prepare your own motion data, please see the `convert_amass_data.py` script in the `scripts` folder. After conversion, you can visually inspect the data with the `vis_motion_mj.py` script.

7. Play the trained policy. In the virtual environment, run:
   ```
   python scripts/train.py --mode play --checkpoint-path <CHECKPOINT PATH>
   ```

   For batch evaluation (e.g., using 4096 envs to evaluate AMASS 11313 motions),
   run:
   ```
   python scripts/train.py --mode eval --checkpoint-path <CHECKPOINT PATH>
   ```

8. Sweep the hyperparameters using CARBS. This is not fully supported yet.
   ```
   python scripts/train.py --mode sweep
   ```

# Notes

- `python scripts/train.py --help` shows the full list of options for the
  environment and training, which allows you to override the defaults in the
  config file.
- I tested the [style discriminator](https://arxiv.org/abs/2104.02180) in the
  original PHC repo and here, saw that it did not improve the imitation
  performance, so turned it off in the config. To enable it, add
  `--env.use-amp-obs
- I saw several times that "fine-tuning" pretrained weights with
  `--checkpoint-path <CHECKPOINT PATH>` resulted in much faster learning. The
  [L2 init reg loss](https://arxiv.org/abs/2308.11958) is being logged in the
  wandb dashboard, and you can see greater L2 distance when learning from
  scratch compared to fine-tuning.
- As mentioned in the [PHC paper](https://arxiv.org/abs/2305.06456), I saw that
  the 6-layer MLP perform better than shallow MLPs, and SiLU activations perform
  better than ReLUs.
- Adding LayerNorm before the last layer tamed the gradient norm, and I
  [swept the hyperparameters](https://wandb.ai/kywch/carbs/sweeps/fupc0sps?nw=nwuserkywch)
  with the max grad norm of 10 (the original PHC repo uses 50).
- Observations are RMS normalized, but the rewards/values are not. The gamma was
  set to 0.98 to manage the value loss. The hyperparameter sweeps consistently
  converged to small lambda values, so I chose 0.2.
- Also, the sweeps consistently converged to very aggressive clip coefs (0.01)
  and higher learning rates. I speculate that since the trainer uses the same
  learning rate for both actor and critic, it's using actor clipping to slow
  down the actor learning relative to the critic.
- I tried using LSTM for both the actor and critic, respectively, and it didn't
  work better. These LSTM policies are included in the code, so feel free to try
  them.

# References

This repository is built on top of the following amazing repositories:

- Main PHC code, motion lib, poselib, and data scripts are from:
  [PHC](https://github.com/ZhengyiLuo/PHC) and
  [Isaac Gym](https://developer.nvidia.com/isaac-gym)
- Refactored PHC and integration with PufferLib:
  [puffer-phc](https://github.com/kywch/puffer-phc) and
- The PPO and CARBS sweep code is from:
  [PufferLib](https://github.com/PufferAI/PufferLib)
- Sample motion data is from:
  [CMU Motion Capture Dataset](http://mocap.cs.cmu.edu/), Subject 5, Motion 6

Please follow the license of the above repositories for usage.

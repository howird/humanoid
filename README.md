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

# References

This repository is built on top of the following amazing repositories:

- `puffer-phc`: Refactored PHC and integration with PufferLib:
  [puffer-phc](https://github.com/kywch/puffer-phc) (based on
  [PHC](https://github.com/zhengyiLuo/phc))
- `hmr2`: Humanoid mesh reconstruction:
  [4D Humans](https://github.com/shubham-goel/4D-Humans)
- `phalp`: Humanoid tracking from appearance:
  [PHALP](https://github.com/brjathu/PHALP)

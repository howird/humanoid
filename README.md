# humanoid
experimenting with controllers for physics-simulated humanoids

## isaacgym

### prerequisites

- Install [pixi](https://pixi.sh/latest/#__tabbed_1_1)
- If using isaacgym:
  - install isaacgym from [here](https://developer.nvidia.com/isaac-gym) (preview 4)
  - extract it to this repository
  - run: `patch -u isaacgym/python/setup.py -i scripts/isaacgym_setup.patch`

## setup

- enter shell with
```bash
pixi shell -e isaacgym
```

## usage

### nix

```bash
nix-shell shell.nix
pixi run -e skillmimic run --test --task SkillMimicBallPlay --num_envs 16 \
  --cfg_env skillmimic/data/cfg/skillmimic.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
  --motion_file skillmimic/data/motions/BallPlay-M/layup \
  --checkpoint skillmimic/data/models/mixedskills/nn/skillmimic_llc.pth
```

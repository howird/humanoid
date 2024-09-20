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

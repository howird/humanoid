from rl_games.algos_torch import torch_ext


def flatten_dict(d, parent_key='', sep='.'):
    flat_dict = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value, new_key, sep=sep))
        else:
            flat_dict[new_key] = value
    return flat_dict


def fix_checkpoint(f, bring_out='running_mean_std', put_in='model', ext='.pth'):
    ckpt = torch_ext.load_checkpoint(f)
    if bring_out not in ckpt:
        return
    bo = {}
    bo[bring_out] = ckpt.pop(bring_out)
    bo = flatten_dict(bo)
    ckpt[put_in].update(bo)
    torch_ext.save_checkpoint(f.replace(ext, '_new'), ckpt)


if __name__ == '__main__':
    import sys
    fix_checkpoint(sys.argv[1])

import logging
import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf


def load_my_state_dict(model: torch.nn.Module, state_dict, lambda_own=lambda x: x):
    own_state = model.state_dict()
    record = {}
    missing_keys, unexpected_keys, mismatch_keys = [], [], []
    for name, param in state_dict.items():
        own_name = lambda_own(name)
        record[own_name] = 0
        if own_name not in own_state:
            unexpected_keys.append(f"{name}->{own_name}")
            logging.warning("Unexpected key from checkpoint %s %s" % (name, own_name))
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if param.size() != own_state[own_name].size():
            logging.warning(
                "size not match %s %s %s"
                % (name, str(param.size()), str(own_state[own_name].size()))
            )
            mismatch_keys.append(own_name)
            continue
        own_state[own_name].copy_(param)

    for n in own_state:
        if n not in record:
            missing_keys.append(n)

    if unexpected_keys:
        logging.warning("Unexpected keys" + str(unexpected_keys))
    if missing_keys:
        logging.warning("Missing keys" + str(missing_keys))
    if mismatch_keys:
        logging.warning("Size mismatched keys" + str(mismatch_keys))
    return missing_keys, unexpected_keys, mismatch_keys


@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg, dataset_cfg, rootdir):
    """Save config files to rootdir."""
    os.makedirs(rootdir, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, "model_config.yaml"))
    with open(os.path.join(rootdir, "dataset_config.yaml"), "w") as f:
        f.write(dataset_cfg.dump())

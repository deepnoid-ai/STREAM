"""Small shared helpers: EMA (adapted from LightningDiT, MIT License), atomic saving, path checks."""

import os
import tempfile
from collections import OrderedDict

import torch


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """ema <- decay * ema + (1 - decay) * model, parameter by parameter."""
    ema_params = OrderedDict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def atomic_torch_save(obj, path):
    """torch.save to a temporary file in the same directory, then os.replace.

    The file gets the default permissions of a newly created file (0666 minus the umask, e.g.
    0644), like a plain torch.save; mkstemp alone would leave it readable by the owner only.
    """
    path = str(path)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        os.close(fd)
        torch.save(obj, tmp_path)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def require_path(path, flag):
    """Fail early with a clear message when an input file or directory does not exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{flag}: '{path}' does not exist.")
    return path

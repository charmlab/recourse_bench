from __future__ import division, print_function

import importlib
import logging
import sys
import types

import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable

log = logging.getLogger(__name__)

suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]


def cprint(color, text, **kwargs):
    del color
    print(text, **kwargs)


def _ensure_package_alias(name):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    return module


def register_checkpoint_aliases():
    for package_name in ("src", "VAE", "VAEAC", "interpret"):
        _ensure_package_alias(package_name)

    module_aliases = {
        "src.utils": "method.clue.library.clue_ml.src.utils",
        "src.radam": "method.clue.library.clue_ml.src.radam",
        "src.probability": "method.clue.library.clue_ml.src.probability",
        "src.gauss_cat": "method.clue.library.clue_ml.src.gauss_cat",
        "src.layers": "method.clue.library.clue_ml.src.layers",
        "VAE.models": "method.clue.library.clue_ml.VAE.models",
        "VAE.fc_gauss_cat": "method.clue.library.clue_ml.AE_models.AE.fc_gauss_cat",
        "VAE.fc_gauss": "method.clue.library.clue_ml.VAE.fc_gauss",
        "VAEAC.models": "method.clue.library.clue_ml.VAEAC.models",
        "VAEAC.fc_gauss_cat": "method.clue.library.clue_ml.VAEAC.fc_gauss_cat",
        "VAEAC.under_net": "method.clue.library.clue_ml.VAEAC.under_net",
        "interpret.generate_data": "method.clue.library.clue_ml.interpret.generate_data",
        "interpret.functionally_grounded": "method.clue.library.clue_ml.interpret.functionally_grounded",
        "interpret.FIDO": "method.clue.library.clue_ml.interpret.FIDO",
    }
    for alias, target in module_aliases.items():
        module = importlib.import_module(target)
        sys.modules[alias] = module
        parent_name, child_name = alias.split(".", 1)
        setattr(sys.modules[parent_name], child_name, module)


def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.0
        i += 1
    f = "%.2f" % nbytes
    return "%s%s" % (f, suffixes[i])


def get_num_batches(nb_samples, batch_size, roundup=True):
    if roundup:
        return (
            nb_samples + (-nb_samples % batch_size)
        ) / batch_size  # roundup division
    else:
        return nb_samples / batch_size


def generate_ind_batch(nb_samples, batch_size, random=True, roundup=True):
    if random:
        ind = np.random.permutation(nb_samples)
    else:
        ind = range(int(nb_samples))
    for i in range(int(get_num_batches(nb_samples, batch_size, roundup))):
        yield ind[i * batch_size : (i + 1) * batch_size]


def to_variable(var=(), cuda=False, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)
        if not v.is_cuda and cuda:
            v = v.cuda()
        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)
        out.append(v)
    return out


class Datafeed(data.Dataset):
    def __init__(self, x_train, y_train=None, transform=None):
        self.data = x_train
        self.targets = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.targets is not None:
            return img, self.targets[index]
        else:
            return img

    def __len__(self):
        return len(self.data)


# ----------------------------------------------------------------------------------------------------------------------
class BaseNet(object):
    def __init__(self):
        log.info("\nNet:")

    def get_nb_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update_lr(self, epoch, gamma=0.99):
        self.epoch += 1
        if self.schedule is not None:
            if len(self.schedule) == 0 or epoch in self.schedule:
                self.lr *= gamma
                log.debug("learning rate: %f  (%d)\n" % (self.lr, epoch))
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr

    def save(self, filename):
        log.info("Writting %s\n" % filename)
        torch.save(
            {
                "epoch": self.epoch,
                "lr": self.lr,
                "model": self.model,
                "optimizer": self.optimizer,
            },
            filename,
        )

    def load(self, filename):
        log.info("Reading %s\n" % filename)
        register_checkpoint_aliases()
        map_location = (
            "cuda"
            if getattr(self, "cuda", False) and torch.cuda.is_available()
            else "cpu"
        )
        state_dict = torch.load(
            filename,
            weights_only=False,
            map_location=map_location,
        )
        self.epoch = state_dict["epoch"]
        self.lr = state_dict["lr"]
        self.model = state_dict["model"]
        self.optimizer = state_dict["optimizer"]
        log.info("restoring epoch: %d, lr: %f" % (self.epoch, self.lr))
        return self.epoch


def torch_onehot(y, Nclass):
    if y.is_cuda:
        y = y.type(torch.cuda.LongTensor)
    else:
        y = y.type(torch.LongTensor)
    y_onehot = torch.zeros((y.shape[0], Nclass)).type(y.type())
    # In your for loop
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot


_map_location = [None]


def MNIST_mean_std_norm(x):
    mean = 0.1307
    std = 0.3081
    x = x - mean
    x = x / std
    return x


class Ln_distance(torch.nn.Module):
    """If dims is None Compute across all dimensions except first"""

    def __init__(self, n, dim=None):
        super(Ln_distance, self).__init__()
        self.n = n
        self.dim = dim

    def forward(self, x, y):
        d = x - y
        if self.dim is None:
            self.dim = list(range(1, len(d.shape)))
        return torch.abs(d).pow(self.n).sum(dim=self.dim).pow(1.0 / float(self.n))


def smooth_median(X, dim=0):
    """Just gets numpy behaviour instead of torch default
    dim is dimension to be reduced, across which median is taken"""
    yt = X.clone()
    ymax = yt.max(dim=dim, keepdim=True)[
        0
    ]  # maybe this is wrong  and dont need keepdim
    yt_exp = torch.cat((yt, ymax), dim=dim)
    smooth_median = (yt_exp.median(dim=dim)[0] + yt.median(dim=dim)[0]) / 2.0
    return smooth_median


class l1_MAD(torch.nn.Module):
    """Intuition behind this metric -> allows variability only where the dataset has variability
    Otherwise it penalises discrepancy heavily. Might not make much sense if data is already normalised to
    unit std. Might also not make sense if we want to detect outlier values in specific features.
    """

    def __init__(self, trainset_data, median_dim=0, dim=None):
        """Median dim are those across whcih to normalise (not features)
        dim is dimension to sum (features)"""
        super(l1_MAD, self).__init__()
        self.dim = dim
        feature_median = smooth_median(trainset_data, dim=median_dim).unsqueeze(
            dim=median_dim
        )
        self.MAD = smooth_median(
            (trainset_data - feature_median).abs(), dim=median_dim
        ).unsqueeze(dim=median_dim)
        self.MAD = self.MAD.clamp(min=1e-4)

    def forward(self, x, y):
        d = x - y
        if self.dim is None:
            self.dim = list(range(1, len(d.shape)))
        return (torch.abs(d) / self.MAD).sum(dim=self.dim)

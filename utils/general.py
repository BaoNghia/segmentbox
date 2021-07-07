import os, re, yaml
import torch
import importlib
import logging
from torch.utils.tensorboard import SummaryWriter
from utils import losses as custom_loss
from pywick import losses as pywick_loss


def get_attr_by_name(func_str):
    """
    Load function by full name
    :param func_str:
    :return: fn, mod
    """
    module_name, func_name = func_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func, module, func_name

def model_loader(config):
    model_dict = config.get('model')
    func, _, _ = get_attr_by_name(model_dict['model.class'])
    # removed_value = model_dict.pop('model.class', 'No Key found')
    return func(**model_dict)


def get_optimizer(config):
    cfg =  config.get("optimizer")
    optimizer_name = cfg["name"]
    try:
        optimizer = getattr(torch.optim, optimizer_name,\
            "The optimizer {} is not available".format(optimizer_name))
    except:
        optimizer = getattr(torch.optim, optimizer_name,\
            "The optimizer {} is not available".format(optimizer_name))
    del cfg['name']
    return optimizer, cfg

def get_lr_scheduler(config):
    cfg = config.get("scheduler")
    scheduler_name = cfg["name"]
    try:
        # if the lr_scheduler comes from torch.optim.lr_scheduler package
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name,\
            "The scheduler {} is not available".format(scheduler_name))
    except:
        # use custom lr_scheduler
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name,\
            "The scheduler {} is not available".format(scheduler_name))
    del cfg['name']
    return scheduler, cfg

def get_loss_fn(config):
    loss_name = config["train"]["loss"]
    try:
        print("Load loss from pywwick")
        # use loss from pywick
        criterion = getattr(pywick_loss, loss_name,\
            "The scheduler {} is not available".format(loss_name))
    except:
        print("Load loss from utils.losses")
        # use custom loss
        criterion = getattr(custom_loss, loss_name,\
            "The scheduler {} is not available".format(loss_name))
    return criterion

def yaml_loader(yaml_file):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.')
    )
    with open(yaml_file) as f:
        config = yaml.load(f, Loader=loader) # cfg dict
    return config


def log_initilize(log_dir):
    log_file = os.path.join(log_dir, "model_logs.txt")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create error file handler and set level to error
    handler = logging.FileHandler(log_file, "a", encoding=None, delay="true")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    handler.terminator = "\n"
    logger.addHandler(handler)
    return logger


def make_writer(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

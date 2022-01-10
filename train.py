from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

# import fasttext
# import fasttext.util

# My Library
from module import MyLightningModel
from dataset import MyDataset
from util.train_util import collate_fn

def get_data_loaders(data_config, utt_list_paths, in_dir, out_dir, collate_fn):
    data_loaders = {}

    for phase in ["train", "dev"]:

        with open(utt_list_paths[phase], "r") as f:
            utts = [l.strip() for l in f if len(l.strip()) > 0]

        in_feats_paths = [
            in_dir / ("-".join(utt.split(":")[:3]) + "-feats.npy") 
            for utt in utts]
        out_feats_paths = [out_dir / in_path.name for in_path in in_feats_paths]

        dataset = MyDataset(in_feats_paths, out_feats_paths)
        data_loaders[phase] = DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=data_config.num_workers,
            shuffle=phase.startswith("train"),
        )

    return data_loaders    

def setup_each_model(
    config, out_dir, fillers, 
    train_filler_rate_list_path, dev_filler_rate_list_path, 
    utt_list_paths, in_feat_dir, out_feat_dir, collate_fn,
    model, fine_tune, max_steps,
    load_ckpt_path=None,
    ):

    # Get filler rate
    train_filler_rate_dict = {}
    dev_filler_rate_dict = {}
    with open(train_filler_rate_list_path, "r") as f:
        for l in f:
            train_filler_rate_dict[l.strip().split(":")[0]] = float(l.strip().split(":")[1])
    with open(dev_filler_rate_list_path, "r") as f:
        for l in f:
            dev_filler_rate_dict[l.strip().split(":")[0]] = float(l.strip().split(":")[1])

    # Get loss weights
    loss_weights = [1 / (train_filler_rate_dict["no_filler"] + train_filler_rate_dict["others"])]
    for filler in fillers:
        if train_filler_rate_dict[filler] == 0:
            loss_weights.append(0)
        else:
            loss_weights.append(1 / train_filler_rate_dict[filler])

    # data loaders
    data_loaders = get_data_loaders(config.data, utt_list_paths, in_feat_dir, out_feat_dir, collate_fn)

    # model
    lr_scheduler_params = config.train.optim.lr_scheduler.params
    lr_scheduler_params["step_size"] = int(config.train.optim.lr_scheduler.params.step_size / len(data_loaders["train"]))
    model_params = {
        "model": model,
        "fillers": fillers,
        "train_filler_rate_dict": train_filler_rate_dict,
        "dev_filler_rate_dict": dev_filler_rate_dict,
        "loss_weights": loss_weights,
        "optimizer_name": config.train.optim.optimizer.name,
        "optimizer_params": config.train.optim.optimizer.params,
        "lr_scheduler_name": config.train.optim.lr_scheduler.name,
        "lr_scheduler_params": lr_scheduler_params,
    }
    if fine_tune:
        pl_model = MyLightningModel.load_from_checkpoint(
            load_ckpt_path,
            **model_params,
        )        
    else:
        pl_model = MyLightningModel(
            **model_params,
        )

    # callbacks
    dirpath = out_dir / config.train.checkpoint.params.dirname
    dirpath.mkdir(parents=True, exist_ok=True)
    ckpt_params = {
        "monitor": config.train.checkpoint.params.monitor,
        "every_n_train_steps": config.train.checkpoint.params.every_n_train_steps,
        "save_top_k": config.train.checkpoint.params.save_top_k,
    }
    ckpt_callback = ModelCheckpoint(
        dirpath=dirpath,
        **ckpt_params,
    )
    lr_monitor = LearningRateMonitor("epoch")
    callbacks = [ckpt_callback, lr_monitor]

    # logging
    loggers = []
    for phase in ["train", "val"]:
        loggers.append(
            pl_loggers.TensorBoardLogger(
                save_dir=str(out_dir),
                name=config.train.logging.name,
                version=config.train.logging.version,
                sub_dir=phase)
        )

    trainer = pl.Trainer(
        # gpu
        gpus=config.train.gpus,
        auto_select_gpus=config.train.auto_select_gpus,
        # training
        max_steps=max_steps,
        check_val_every_n_epoch=config.train.check_val_every_n_epoch,
        gradient_clip_val=config.train.gradient_clip_val,
        # checkpoint/logging
        default_root_dir=out_dir,
        callbacks=callbacks,
        logger=loggers,
        profiler="simple",
    )

    return data_loaders, pl_model, trainer


@hydra.main(config_path="conf/train", config_name="config")
def myapp(config: DictConfig):

    # Set output directory
    out_dir = Path(to_absolute_path(config.train.out_dir))
    exist_ok = True if config.train.resume else False
    out_dir.mkdir(parents=True, exist_ok=exist_ok)

    # Save config
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Load config
    preprocess_config = OmegaConf.load(Path(config.data.preprocessed_dir) / "config.yaml")

    # Get cluster information
    group_persons_dict = {}
    with open(preprocess_config.group_list_path, "r") as f:
        for l in f:
            i_class = int(l.strip().split(":")[1])
            if i_class in group_persons_dict.keys():
                group_persons_dict[i_class].append(l.strip().split(":")[0])
            else:
                group_persons_dict[i_class] = [l.strip().split(":")[0]]
    n_cluster = len(group_persons_dict.keys())

    # Random seed
    pl.seed_everything(config.random_seed)

    # Fillers
    filler_list_path = Path(to_absolute_path(config.data.filler_list))
    with open(filler_list_path, "r") as f:
        fillers = [l.strip() for l in f]

    # Set feature directory
    in_feat_dir = Path(to_absolute_path(config.data.in_dir))
    out_feat_dir = Path(to_absolute_path(config.data.out_dir))

    # Set model parameters
    model = hydra.utils.instantiate(config.model.netG)

    # Trainig non-personalized model
    out_dir_m = out_dir / "non_personalized"
    train_filler_rate_list_path = config.data.preprocessed_dir / "train_all_filler_rate.list"
    dev_filler_rate_list_path = config.data.preprocessed_dir / "dev_all_filler_rate.list"
    utt_list_paths = {}
    utt_list_paths["train"] = config.data.preprocessed_dir / "train_all.list"
    utt_list_paths["dev"] = config.data.preprocessed_dir / "dev_all.list"
    fine_tune = False
    max_steps = 60000
    data_loaders, pl_model, trainer = setup_each_model(
        config, out_dir_m, fillers, 
        train_filler_rate_list_path, dev_filler_rate_list_path, 
        utt_list_paths, in_feat_dir, out_feat_dir, collate_fn,
        model, fine_tune, max_steps,
        )
    ckpt_path = config.train.load_ckpt_path if config.train.resume else None
    trainer.fit(pl_model, data_loaders["train"], data_loaders["dev"], ckpt_path=ckpt_path)

    # Trainig group-dependent models
    for i in range(1, 1+n_cluster):
        out_dir_m = out_dir / "group{}".format(str(i))
        train_filler_rate_list_path = \
            config.data.preprocessed_dir / "train_group{}_filler_rate.list".format(str(i))
        dev_filler_rate_list_path = \
            config.data.preprocessed_dir / "dev_group{}_filler_rate.list".format(str(i))
        utt_list_paths = {}
        utt_list_paths["train"] = config.data.preprocessed_dir / "train_group{}.list".format(str(i))
        utt_list_paths["dev"] = config.data.preprocessed_dir / "dev_group{}.list".format(str(i))
        fine_tune = True
        max_steps = 10000
        data_loaders, pl_model, trainer = setup_each_model(
            config, out_dir_m, fillers, 
            train_filler_rate_list_path, dev_filler_rate_list_path, 
            utt_list_paths, in_feat_dir, out_feat_dir, collate_fn,
            model, fine_tune, max_steps,
            )
        ckpt_path = config.train.load_ckpt_path if config.train.resume else None
        trainer.fit(pl_model, data_loaders["train"], data_loaders["dev"], ckpt_path=ckpt_path)

if __name__=="__main__":
    myapp()
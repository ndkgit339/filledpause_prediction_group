from pathlib import Path
from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# My library
import fp_pred_group.model
from fp_pred_group.dataset import MyDataset
from fp_pred_group.module import MyLightningModel
from fp_pred_group.util.train_util import collate_fn

def predict(utt_list_path, in_feat_dir, out_feat_dir, out_dir,
            batch_size, num_workers, trainer, model, fp_list):

    # Load utt list
    with open(utt_list_path, "r") as f:
        sentence_list = [l.strip() for l in f if len(l.strip()) > 0]
        utt_name_list = [sen.split(":")[0] for sen in sentence_list]

    # Dataset
    in_feat_dir = Path(in_feat_dir)
    out_feat_dir = Path(out_feat_dir)

    in_feats_paths = [
        p for p in in_feat_dir.glob("*-feats.npy") 
        if p.stem.split("-feats")[0] in utt_name_list]
    out_feats_paths = [
        out_feat_dir / in_path.name for in_path in in_feats_paths]

    dataset = MyDataset(in_feats_paths, out_feats_paths)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn,
                             pin_memory=True,
                             num_workers=num_workers,
                             shuffle=False)

    # Prediction
    out_utt_list = []
    outputs = trainer.predict(model, data_loader)
    for output in tqdm(outputs):
        batch_idx = output["batch_idx"]
        predictions = output["predictions"]

        for in_feats_path, prediction in zip(
                in_feats_paths[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size],
                predictions):

            utt_id = in_feats_path.stem.split("-feats")[0]
            for sen in sentence_list:
                if sen.startswith(f"{utt_id}"):
                    utt_text = sen.split(":")[-1]

            utt_wo_fps = [
                w for w in utt_text.split(" ") 
                if not w.startswith("(F")]
            predicted_fp_tags = torch.argmax(prediction, dim=1)

            # print(prediction.shape)
            # print(utt_wo_fps)
            # print(len(utt_wo_fps), len(predicted_fp_tags))

            predicted_text = []
            if int(predicted_fp_tags[0]) > 0:
                predicted_text.append(
                    "(F{})".format(
                        fp_list[int(predicted_fp_tags[0]) - 1]))
            for i in range(len(utt_wo_fps)):
                predicted_text.append(utt_wo_fps[i])
                if int(predicted_fp_tags[i + 1]) > 0:
                    predicted_text.append(
                        "(F{})".format(
                            fp_list[int(predicted_fp_tags[i + 1]) - 1]))

            outtexts = [
                "{}:".format(utt_id), 
                "\ttarget text: \t{}".format(utt_text),
                "\tpredicted text: \t{}".format(" ".join(predicted_text)),
            ]

            out_utt_list.append(outtexts)

    out_utt_list = [
        "\n".join(outtexts) 
        for outtexts in sorted(out_utt_list, key=lambda x: x[0])]

    with open(out_dir / "fp_prediction.txt", "w") as f:
        print("writing prediction...")
        f.write("\n".join(out_utt_list))

@hydra.main(config_path="conf/predict", config_name="config")
def main(config: DictConfig):

    # Phase
    phase = "pred"

    # Input directory
    in_feat_dir = Path(config.data.data_dir) / "infeats"
    out_feat_dir = Path(config.data.data_dir) / "outfeats"

    # Out directory
    exp_dir = Path(to_absolute_path(config[phase].exp_dir))
    ckpt_dir = exp_dir / "ckpt"
    out_dir = Path(to_absolute_path(config[phase].out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    train_config = OmegaConf.load(exp_dir / "config.yaml")

    # Save config
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(config, f)

    # Set random seed
    pl.seed_everything(config.random_seed)

    # FPs
    fp_list_path = Path(to_absolute_path(config.data.fp_list))
    with open(fp_list_path, "r") as f:
        fp_list = [l.strip() for l in f]

    # Load model
    model = hydra.utils.instantiate(train_config.model.netG)
    ckpt_path = list(ckpt_dir.glob(
        "*-step={}.ckpt".format(str(config[phase].checkpoint.step))
    ))[0]
    pl_model = MyLightningModel.load_from_checkpoint(
        str(ckpt_path),
        model=model,
        fp_list=fp_list,
        strict=False)

    # Trainer
    trainer = pl.Trainer(
        gpus=config[phase].gpus,
        auto_select_gpus=config[phase].auto_select_gpus,
        default_root_dir=exp_dir)

    # Predict
    predict(config.data.utt_list, 
            in_feat_dir, out_feat_dir, out_dir,
            config.data.batch_size, config.data.num_workers,
            trainer, pl_model, fp_list)

if __name__=="__main__":
    main()
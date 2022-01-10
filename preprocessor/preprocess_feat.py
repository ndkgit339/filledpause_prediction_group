import time
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

# 言語処理
# import fasttext
# import fasttext.util
from transformers import BertTokenizer, BertModel

# データ処理
import numpy as np
import torch

def extract_feats(config):
    start = time.time()

    # Fillers
    with open(config.filler_list_path, "r") as f:
        fillers = [l.strip() for l in f]

    # if config.nlp_model == "fasttext":
    #     ft = fasttext.load_model(str(Path(config.fasttext_dir) / f"cc.ja.{config.fasttext_dim}.bin"))
    #     def preprocess_ipu(speaker_id, koen_id, ipu_id, ipu_tagtext, in_dir, out_dir):

    #         word_embeds = [np.array([0] * 300)]
    #         filler_labels = [0]     # 呼気段落の頭の場合がある
    #         for m in ipu_tagtext.split(" "):
    #             if m.startswith("(F"):
    #                 filler = m.split("(F")[1].split(")")[0]
    #                 if filler in fillers:
    #                     filler_labels[-1] = fillers.index(filler) + 1

    #             elif m != "":
    #                 word_embeds.append(ft.get_word_vector(m))
    #                 filler_labels.append(0)
            
    #         np.save(in_dir / f"{speaker_id}-{koen_id}-{ipu_id}-feats.npy", np.array(word_embeds))
    #         np.save(out_dir / f"{speaker_id}-{koen_id}-{ipu_id}-feats.npy", np.array(filler_labels))
    # elif config.nlp_model == "bert":

    # Prepare bert
    bert_model_dir = Path(config.bert_model_dir)
    vocab_file_path = bert_model_dir / "vocab.txt"
    bert_tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, do_basic_tokenize=False)
    bert_model = BertModel.from_pretrained(bert_model_dir)
    bert_model.eval()
    def preprocess_ipu(speaker_id, koen_id, ipu_id, ipu_tagtext, in_dir, out_dir):

        # get tokens and filler labels
        filler_labels = [0]     # fillers sometimes appear at the head of the breath group
        tokens = ["[CLS]"]
        for m in ipu_tagtext.split(" "):
            if m.startswith("(F"):
                filler = m.split("(F")[1].split(")")[0]
                if filler in fillers:
                    filler_labels[-1] = fillers.index(filler) + 1
            elif m != "":
                tokens.append(m)
                filler_labels.append(0)

        tokens += ["[SEP]"]
        filler_labels.append(0)

        # get embedding
        token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        token_tensor = torch.Tensor(token_ids).unsqueeze(0).to(torch.long)
        outputs = bert_model(token_tensor)
        outputs_numpy = outputs[0].numpy().squeeze(axis=0).copy()
        
        assert outputs_numpy.shape[0] == np.array(filler_labels).shape[0], "1st array length {} should be equal to 2nd array length {}".format(outputs_numpy.shape[0], np.array(filler_labels).shape[0])
        np.save(in_dir / f"{speaker_id}-{koen_id}-{ipu_id}-feats.npy", outputs_numpy)
        np.save(out_dir / f"{speaker_id}-{koen_id}-{ipu_id}-feats.npy", np.array(filler_labels))

    # extraxt features
    infeats_dir = Path(config.out_dir) / "infeats"
    outfeats_dir = Path(config.out_dir) / "outfeats"
    infeats_dir.mkdir(parents=True, exist_ok=True)
    outfeats_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(config.out_dir) / f"ipu.list", "r") as f:
        ipus = [tuple(l.split(":")) for l in f.readlines()]
    with torch.no_grad():
        for speaker_id, koen_id, ipu_id, ipu in tqdm(ipus):
            preprocess_ipu(speaker_id, koen_id, ipu_id, ipu, infeats_dir, outfeats_dir)

    # count time
    n_ipu = len(ipus)
    elapsed_time = time.time() - start
    time_log ="elapsed_time of feature extraction: {} [sec]".format(elapsed_time)
    time_log_ipu ="elapsed_time of feature extraction (per IPU): {} [sec]".format(elapsed_time / n_ipu)
    print(time_log + "\n" + time_log_ipu)
    with open(Path(config.out_dir) / "time.log", "w") as f:
        f.write(time_log + "\n" + time_log_ipu)

@hydra.main(config_path="conf/preprocess", config_name="config")
def myapp(config: DictConfig):
    extract_feats(config)
    
if __name__=="__main__":
    myapp()
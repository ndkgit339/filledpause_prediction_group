import re
from pathlib import Path

import hydra
from omegaconf import DictConfig

def analyze_filler(config):

    # Fillers
    with open(config.filler_list_path, "r") as f:
        fillers = [l.strip() for l in f]

    # Get frequency rate of each filler word
    for tagtext_list_path in Path(config.out_dir).glob("*.list"):
        with open(tagtext_list_path, "r") as f:
            tagtext_all = f.read()

        n_position = 0
        for tag_text in tagtext_all.split("\n"):
            if re.fullmatch(r".*?:.*?:.*", tag_text):
                tag_text = tag_text.split(":")[2]
                tag_text = re.sub(r"\(F.*?\)", "", tag_text)
                n_morph = 0
                for t in tag_text.split(" "):
                    if t != "":
                        n_morph += 1
                n_position += n_morph + 1

        n_each_filler_dict = {}
        n_filler = 0
        for filler in fillers:
            n = tagtext_all.count(f"(F{filler})")
            n_filler += n
            n_each_filler_dict[filler] = n / n_position

        n_filler_all = len(re.findall(r"\(F.*?\)", tagtext_all))
        n_each_filler_dict["others"] = (n_filler_all - n_filler) / n_position
        n_each_filler_dict["no_filler"] = 1 - n_filler_all / n_position


        n_each_filler_text = "\n".join(
            [f"{filler}:{n}" for filler, n in n_each_filler_dict.items()]
        )
        
        with open(tagtext_list_path.parent / (tagtext_list_path.stem + "_filler_rate.list"), "w") as f:
            f.write(n_each_filler_text)

@hydra.main(config_path="conf/preprocess", config_name="config")
def myapp(config: DictConfig):
    analyze_filler(config)

if __name__=="__main__":
    myapp()
import os

def get_file_path(cfg):
    parent_dir= cfg["trainer"]["dataset"]["path"].replace("/","-")
    sub_parent_dir= cfg["trainer"]["dataset"]["name"]
    sub_sub_parent_dir = f'{cfg["model"]["tokenizer"]["model_name"].split("/")[-1] if cfg["model"]["tokenizer"]["type"] == "huggingface" else cfg["model"]["tokenizer"]["type"]}'
    file_path = os.path.join(
        cfg["general"]["paths"]["data_dir"],
        parent_dir,
        sub_parent_dir,
        sub_sub_parent_dir
    )
    
    return file_path

import os

def get_file_path(cfg):
    # dataset_names = cfg["trainer"]["dataset"]
    # if isinstance(dataset_names, str):
    #     dataset_names = [dataset_names]
    # parent_dir = "_".join(dataset_names)
    # sub_parent_dir =  f'{cfg["tokenizer"]["model_name"].split("/")[-1] if cfg["tokenizer"]["type"] == "huggingface" else cfg["tokenizer"]["type"]}'
    # file_path = os.path.join(
    #     cfg["general"]["paths"]["data_dir"],
    #     parent_dir,
    #     sub_parent_dir
    # )

    parent_dir= cfg["trainer"]["dataset"]["path"]
    sub_parent_dir= cfg["trainer"]["dataset"]["name"]
    sub_sub_parent_dir = f'{cfg["model"]["tokenizer"]["model_name"].split("/")[-1] if cfg["model"]["tokenizer"]["type"] == "huggingface" else cfg["model"]["tokenizer"]["type"]}'
    file_path = os.path.join(
        cfg["general"]["paths"]["data_dir"],
        parent_dir,
        sub_parent_dir,
        sub_sub_parent_dir
    )
    
    return file_path

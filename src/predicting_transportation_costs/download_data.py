import requests
from predicting_transportation_costs import config as cfg
from huggingface_hub import hf_hub_download

def main() -> None:
    download_file("train-test.csv")
    download_file("validation.csv")

def download_file(file_name):

    file= cfg.DATA_DIR / file_name
    if file.exists():
        print(f"{file} already exists. Skipping download.")
        return
     
    hf_hub_download(
        repo_id="KuroEinzbern/transportation_costs",
        filename=str(file_name),
        repo_type="dataset",
        local_dir=cfg.DATA_DIR,
    )

if __name__ == "__main__":
    main()
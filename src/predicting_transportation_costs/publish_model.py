import os
from huggingface_hub import HfApi
import predicting_transportation_costs.config as cfg
from dotenv import load_dotenv

load_dotenv()


MODEL_VERSION = os.environ["MODEL_VERSION"]
MODEL_PATH = cfg.MODEL_DIR / f"model_{MODEL_VERSION}"

api = HfApi()



def main():
    api.upload_file(
    path_or_fileobj=str(MODEL_PATH),
    path_in_repo=MODEL_PATH.name,
    repo_id="KuroEinzbern/transportation_cost",
    repo_type="model",  
    )






if __name__ == "__main__":
    main()
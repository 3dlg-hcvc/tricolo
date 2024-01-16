import hydra
import pickle
from tricolo.evaluation import compute_metrics


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    with open(cfg.prediction_file_path, "rb") as f:
        embeddings_dict = pickle.load(f)

    _ = compute_metrics(cfg.data.dataset, embeddings_dict, metric=cfg.evaluation.metric, print_results=True)


if __name__ == "__main__":
    main()

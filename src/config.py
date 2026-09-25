"""Central configuration: paths, class names, and default hyper-parameters.

Every later day (classic aug, Mixup/SMOTE, GAN) imports from here so that all
four configurations share *exactly* the same data, split, image size, and
training recipe. Changing a value here changes it for every experiment.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Dataset constants (HAM10000 == ISIC 2018 Task 3 training set, 10,015 images)
# --------------------------------------------------------------------------- #
# Column order of the official ground-truth CSV.
CLASS_CODES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AKIEC": "Actinic keratosis / Bowen's",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
}
NUM_CLASSES = len(CLASS_CODES)

ISIC_BASE_URL = "https://isic-challenge-data.s3.amazonaws.com/2018"
DOWNLOADS = {
    # name on disk               : URL
    "ISIC2018_Task3_Training_Input.zip": f"{ISIC_BASE_URL}/ISIC2018_Task3_Training_Input.zip",  # ~2.6 GB images
    "ISIC2018_Task3_Training_GroundTruth.zip": f"{ISIC_BASE_URL}/ISIC2018_Task3_Training_GroundTruth.zip",  # labels
    "ISIC2018_Task3_Training_LesionGroupings.csv": f"{ISIC_BASE_URL}/ISIC2018_Task3_Training_LesionGroupings.csv",  # lesion_id
}

# ImageNet statistics (we fine-tune ImageNet-pretrained weights).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Rare-class selection rule (blueprint: "well under 2% of total examples").
RARE_SHARE_THRESHOLD = 0.02


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    """All filesystem locations used by the project.

    project_dir : the repo folder (on Colab: inside Google Drive so results persist)
    raw_dir     : where the 2.6 GB zip is downloaded/extracted. On Colab this should
                  be LOCAL disk (/content/...) — much faster than Drive and it does
                  not eat your Drive quota. Only the small resized cache goes to Drive.
    """

    project_dir: Path
    raw_dir: Path

    def __post_init__(self):
        self.project_dir = Path(self.project_dir)
        self.raw_dir = Path(self.raw_dir)

    # --- raw data ---------------------------------------------------------- #
    @property
    def raw_images_dir(self) -> Path:
        return self.raw_dir / "ISIC2018_Task3_Training_Input"

    @property
    def raw_labels_csv(self) -> Path:
        return self.raw_dir / "ISIC2018_Task3_Training_GroundTruth" / "ISIC2018_Task3_Training_GroundTruth.csv"

    @property
    def raw_lesion_csv(self) -> Path:
        return self.raw_dir / "ISIC2018_Task3_Training_LesionGroupings.csv"

    # --- processed data (persisted) --------------------------------------- #
    @property
    def processed_dir(self) -> Path:
        return self.project_dir / "data" / "processed"

    @property
    def metadata_csv(self) -> Path:
        return self.processed_dir / "metadata.csv"

    @property
    def splits_dir(self) -> Path:
        return self.project_dir / "data" / "splits"

    def split_csv(self, seed: int) -> Path:
        return self.splits_dir / f"split_seed{seed}.csv"

    def image_cache(self, size: int) -> Path:
        return self.processed_dir / f"images_{size}px_uint8.npy"

    @property
    def dataset_info_json(self) -> Path:
        return self.processed_dir / "dataset_info.json"

    # --- results ----------------------------------------------------------- #
    @property
    def results_dir(self) -> Path:
        return self.project_dir / "results"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def runs_dir(self) -> Path:
        return self.results_dir / "runs"

    @property
    def results_log(self) -> Path:
        return self.results_dir / "results_log.csv"

    def make_dirs(self) -> None:
        for d in [self.raw_dir, self.processed_dir, self.splits_dir,
                  self.figures_dir, self.runs_dir]:
            d.mkdir(parents=True, exist_ok=True)


def default_paths() -> Paths:
    """Colab-friendly defaults, overridable by environment variables."""
    in_colab = "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists()
    project = os.environ.get(
        "RDA_PROJECT_DIR",
        "/content/drive/MyDrive/rare-disease-aug" if in_colab else str(Path(__file__).resolve().parents[1]),
    )
    raw = os.environ.get("RDA_RAW_DIR", "/content/ham10000_raw" if in_colab else str(Path(project) / "data" / "raw"))
    return Paths(Path(project), Path(raw))


# --------------------------------------------------------------------------- #
# Training configuration (shared by every experiment for a fair comparison)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    run_name: str = "day1_baseline_noaug"
    config_name: str = "baseline_no_aug"   # the experiment arm, used in the results table
    arch: str = "resnet18"
    pretrained: bool = True                # ImageNet transfer learning
    image_size: int = 224
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-4
    # Windows starts DataLoader workers by copying the dataset (incl. the 1.5 GB image array) into
    # each worker, which is slow and memory-hungry; the images are already in RAM, so 0 workers is fastest there.
    num_workers: int = 0 if os.name == "nt" else 2
    seed: int = 42                         # training seed (varied on Days 6-7)
    split_seed: int = 42                   # data-split seed (kept FIXED for all experiments)
    amp: bool = True                       # mixed precision on GPU
    select_metric: str = "val_macro_f1"    # checkpoint selection — same rule for every config
    rare_class: str = "DF"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

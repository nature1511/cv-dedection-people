# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/ssd/entrypoints.py
import urllib.request
from pathlib import Path

import torch

from detection.config.config import Configs


def nvidia_ssd(
    pretrained_default=True,
    pretrainded_custom=False,
    path=Configs.path_weight_model,
    device=Configs.device,
    label_num=Configs.model_number_classes,
):
    """Constructs an SSD300 model."""
    from . import model as ssd

    model = ssd.SSD300(label_num=label_num)
    if torch.cuda.is_available() and device == "cuda":
        model = model.cuda()

    if isinstance(path, str):
        path = Path(path)

    path.mkdir(parents=True, exist_ok=True)

    if pretrained_default or pretrainded_custom:
        if pretrained_default:
            checkpoint = "https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/20.06.0/files/nvidia_ssdpyt_amp_200703.pt"
            model_name = str(Path(checkpoint).name)
            path_to_model = path / model_name
            if not path_to_model.is_file():
                print("---loading model weights and saving to ssd folder--- ")
                urllib.request.urlretrieve(checkpoint, path_to_model)
                print("---loading model complete--- ")

        elif pretrainded_custom:
            path_to_model = path

        if not torch.cuda.is_available() or device == "cpu":
            ckpt = torch.load(path_to_model, map_location=torch.device("cpu"))
        elif device == "cuda" and torch.cuda.is_available():
            ckpt = torch.load(path_to_model)

        ckpt = ckpt["model"]
        model.load_state_dict(ckpt)
    return model

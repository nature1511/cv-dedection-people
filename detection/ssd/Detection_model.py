from pathlib import Path

from detection.config.config import Configs
from detection.ssd.create_model import nvidia_ssd


class Detection_model:
    def __init__(
        self,
        work_directory=Path().cwd(),
        device=Configs.device,
        label_num=Configs.model_number_classes,
        pretrained_default=True,
        pretrainded_custom=False,
        path_weight_model=Configs.path_weight_model,
        use_head=Configs.use_head,
    ):
        self.work_directory = work_directory
        self.device = device
        self.label_num = label_num
        self.pretrained_default = pretrained_default
        self.pretrainded_custom = pretrainded_custom
        model = nvidia_ssd(
            pretrained_default=self.pretrained_default,
            pretrainded_custom=self.pretrainded_custom,
            path=self.work_directory / path_weight_model,
            device=self.device,
            label_num=self.label_num,
        )
        self.info = {1: "person", 2: "head"}
        self.model = model
        self.weight = "default"
        self.use_head = use_head

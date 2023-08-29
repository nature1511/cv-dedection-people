from pathlib import Path

import torch
from tqdm import tqdm

from config.config import Configs
from ssd.dataloader import ImagesDataset
from ssd.decode_results import Processing as processing
from utils.utils import get_bboxes


def detect_images(
    model,
    device=Configs.device,
    path_to_data=Configs.path_data,
    criteria_iou=Configs.decode_result["criteria"],
    max_output_iou=Configs.decode_result["max_output"],
    prob_threshold=Configs.decode_result["pic_threshold"],
    use_head=Configs.use_head,
):
    print("run detect images")
    model.eval()
    if isinstance(path_to_data, str):
        path_to_data = Path(path_to_data)

    data = ImagesDataset(
        path=path_to_data,
        device=device,
    )

    if len(data.images) == 0:
        return "no images found!"
    ans = {}
    for image, file in tqdm(data):
        image = image.unsqueeze(0)
        if device == "cuda" and torch.cuda.is_available():
            image = image.cuda()

        with torch.no_grad():
            detections = model(image)

        results_per_input = processing.decode_results(
            predictions=detections,
            criteria=criteria_iou,
            max_output=max_output_iou,
        )

        best_results_per_input = processing.pick_best(
            detections=results_per_input[0],
            threshold=prob_threshold,
        )
        ans[file] = get_bboxes(
            prediction=best_results_per_input, original=file, use_head=use_head
        )

    return ans

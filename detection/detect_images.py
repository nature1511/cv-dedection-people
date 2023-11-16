from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from detection.config.config import Configs
from detection.ssd.dataloader import ImagesDataset
from detection.ssd.decode_results import Processing as processing
from detection.utils.utils import get_bboxes


def detect_images_from_folder(
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


def detect_image_batch(
    model,
    images,
    device=Configs.device,
    criteria_iou=Configs.decode_result["criteria"],
    max_output_iou=Configs.decode_result["max_output"],
    prob_threshold=Configs.decode_result["pic_threshold"],
    use_padding_in_transform=Configs.use_padding_in_image_transform,
    use_head=Configs.use_head,
):
    model.eval()
    inputs = []
    for image in images:
        img1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img1, "RGB")
        inputs.append(img)

    transform = transforms.Compose(
        [
            # SquarePad(),
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ]
    )
    if len(inputs) == 1:
        tensor = transform(inputs[0]).unsqueeze(0)
    else:
        tensor = torch.stack([transform(img) for img in inputs])

    if device == "cuda" and torch.cuda.is_available():
        tensor = tensor.cuda()

    with torch.no_grad():
        detections = model(tensor)

    results_per_inputs = processing.decode_results(
        predictions=detections,
        criteria=criteria_iou,
        max_output=max_output_iou,
    )

    ans = {}
    for i, results_per_input in enumerate(results_per_inputs):
        best_results_per_input = processing.pick_best(
            detections=results_per_input,
            threshold=prob_threshold,
        )
        ans["img_num_" + str(i)] = get_bboxes(
            prediction=best_results_per_input,
            original=images[i],
            use_padding=use_padding_in_transform,
            use_head=use_head,
        )

    return ans

from pathlib import Path

import torch

from detect_images_from_folder import detect_images
from detect_video import detect_video
from ssd.create_model import nvidia_ssd
from utils.utils import draw_bboxes_and_save_image, draw_boxes_and_save_video


def main():
    torch.cuda.empty_cache()
    work_directory = Path().cwd()

    model = nvidia_ssd(
        pretrained_default=False,
        pretrainded_custom=True,
        path=work_directory / "weight\\state best_model_at_adam2.pth",
        device="cuda",
        label_num=3,
    )
    res_img = detect_images(
        model=model,
        device="cuda",
        path_to_data=work_directory / "data",
        prob_threshold=0.3,
        use_head=True,
    )
    draw_bboxes_and_save_image(res_img, use_head=True)

    res_video = detect_video(
        model=model,
        device="cuda",
        path_to_data=work_directory / "data",
        prob_threshold=0.3,
        use_head=True,
    )
    draw_boxes_and_save_video(res_video, use_head=True)


if __name__ == "__main__":
    main()

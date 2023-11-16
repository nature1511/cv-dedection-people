import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm

from detection.config.config import Configs


def set_seed(seed=Configs.random_seed):
    """Делает наши результаты воспроизводимыми (вычисления могут немного больше времени занимать)"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


def get_bboxes(
    prediction,
    original,
    use_padding=Configs.use_padding_in_image_transform,
    use_head=Configs.use_head,
):
    if isinstance(original, str):
        original = cv2.imread(original)

    elif isinstance(original, np.ndarray):
        original = original.copy()

    else:
        original = cv2.imread(str(original))

    original_shape = (original.shape[0], original.shape[1])
    bboxes, classes, _ = prediction
    info = {1: "person", 2: "head"}

    if use_padding:
        orig_h = max(original.shape[0], original.shape[1])
        orig_w = orig_h
        # calculate shapes for padding
        new_shape = (300, 300)
        ratio = float(max(new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])
        new_h, new_w = new_size
        delta_h = 300 - new_h
        delta_w = 300 - new_w

    else:
        orig_h, orig_w = original.shape[0], original.shape[1]
        delta_h = 0
        delta_w = 0

    info = {1: "person", 2: "head"}
    bbx_detection = {}
    bbx_detection["person"] = []
    if use_head:
        bbx_detection["head"] = []

    for idx, bbox in enumerate(bboxes):
        if not use_head and (classes[idx] != 1):
            continue
        x1, y1, x2, y2 = bbox
        # resize the bounding boxes from the normalized to 300 pixels
        x1, y1 = int(x1 * 300) - delta_w // 2, int(y1 * 300) - delta_h // 2
        x2, y2 = int(x2 * 300) - delta_w // 2, int(y2 * 300) - delta_h // 2
        # resizing again to match the original dimensions of the image
        x1, y1 = int((x1 / 300) * orig_w), int((y1 / 300) * orig_h)
        x2, y2 = int((x2 / 300) * orig_w), int((y2 / 300) * orig_h)
        # draw the bounding boxes around the objects
        class_name = info[classes[idx]]
        bbx_detection[class_name].append([x1, y1, x2, y2])

    return bbx_detection


def draw_bboxes_one_image(
    prediction,
    original,
    use_head=Configs.use_head,
):
    if isinstance(original, str):
        original = cv2.imread(original)

    elif isinstance(original, np.ndarray):
        original = original.copy()

    else:
        original = cv2.imread(str(original))

    info = {"person": (0, 0, 255), "head": (0, 255, 0)}

    bbx_ps = prediction["person"]
    for bbx_p in bbx_ps:
        x1, y1, x2, y2 = bbx_p

        cv2.rectangle(
            original, (x1, y1), (x2, y2), info["person"], 2, cv2.LINE_AA
        )
        cv2.putText(
            original,
            "person",
            (x1, y1 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            info["person"],
            1,
        )
    if use_head and prediction["head"]:
        bbx_hs = prediction["head"]
        for bbx_h in bbx_hs:
            x1, y1, x2, y2 = bbx_h
            cv2.rectangle(
                original, (x1, y1), (x2, y2), info["head"], 2, cv2.LINE_AA
            )
            cv2.putText(
                original,
                "head",
                (x1, y1 + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                info["head"],
                1,
            )

    return original


def draw_bboxes_and_save_image(
    detect_res,
    path_new_data=Configs.path_new_data,
    use_head=Configs.use_head,
    save_image=False,
    show_image=False,
):
    if detect_res == "no images found!":
        print("no images found!")
        return "no images found!"
    if save_image:
        if isinstance(path_new_data, str):
            path_new_data = Path(path_new_data)

        path_new_data.mkdir(parents=True, exist_ok=True)

    for img in detect_res.keys():
        new_image = draw_bboxes_one_image(
            prediction=detect_res[img], original=img, use_head=use_head
        )

        if isinstance(img, str):
            img = Path(img)
        orginal_name = img.name

        if save_image:
            path_save_image = path_new_data / ("new_" + orginal_name)
            cv2.imwrite(str(path_save_image), new_image)
        if show_image:
            cv2.imshow("1", new_image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


def draw_boxes_and_save_video(
    detect_res, path_new_data=Configs.path_new_data, use_head=Configs.use_head
):
    if detect_res == "no video found":
        print("no video found")
        return "no video found"
    if isinstance(path_new_data, str):
        path_new_data = Path(path_new_data)
    path_new_data.mkdir(parents=True, exist_ok=True)
    print("run print bbx on video")
    batch_size = detect_res["batch_size"]
    videos = [video for video in detect_res.keys() if video != "batch_size"]
    for video in videos:
        if isinstance(video, str):
            video = Path(video)
        orginal_name = video.name
        orginal_name = orginal_name.rsplit(".", 1)[0]
        path_save_video = path_new_data / ("new_" + orginal_name + ".avi")

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)  # fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            str(path_save_video), fourcc, fps, (f_width, f_height)
        )
        batch_number = total_frames // batch_size + int(
            (total_frames % batch_size) > 0
        )
        pbar = tqdm(
            total=batch_number, desc="extracting frames at fps: {}".format(fps)
        )

        count = 0
        batch = []
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            batch.append(image)

            if (len(batch) == batch_size) or (
                (total_frames % batch_size == len(batch))
                and (count == batch_number - 1)
            ):
                # get images for batch
                images = detect_res[str(video)]["batch_" + str(count)]
                new_images = []
                for i, img in enumerate(images):
                    bbx = images[img]
                    new_image = draw_bboxes_one_image(
                        prediction=bbx, original=batch[i], use_head=use_head
                    )
                    new_images.append(new_image)

                for new_image in new_images:
                    out.write(new_image)

                batch = []
                count += 1
                pbar.update(1)

        cap.release()
        out.release()
        pbar.close()
        cv2.destroyAllWindows()


def save_model(
    model,
    optimizer=None,
    model_name="model_name",
    path="weight",
    lr_scheduler=None,
):
    if isinstance(path, str):
        path = Path(path)
    path_save_state = path / ("state " + model_name + ".pth")
    state = {
        "model_name": model_name,
        "model": model.state_dict(),
        "optimizer_state": None
        if optimizer is None
        else optimizer.state_dict(),
        "lr_scheduler_state": None
        if lr_scheduler is None
        else lr_scheduler.state_dict(),
    }
    torch.save(state, path_save_state)

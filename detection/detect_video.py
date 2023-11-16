from pathlib import Path

import cv2
from tqdm import tqdm

from detection.config.config import Configs
from detection.detect_images import detect_image_batch


def detect_one_video(
    model,
    video,
    device=Configs.device,
    batch_size=Configs.batch_size,
    criteria_iou=Configs.decode_result["criteria"],
    max_output_iou=Configs.decode_result["max_output"],
    prob_threshold=Configs.decode_result["pic_threshold"],
    use_head=Configs.use_head,
):
    model.eval()

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)  # fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    batch_number = total_frames // batch_size + \
        int((total_frames % batch_size) > 0)

    pbar = tqdm(
        total=batch_number,
        desc="extracting frames at fps: {}".format(fps),
    )
    count = 0
    batch = []
    bbx_batch = {}
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        batch.append(image)

        if (len(batch) == batch_size) or (
            (total_frames % batch_size == len(batch)) and (
                count == batch_number - 1)
        ):
            bbx_batch["batch_" + str(count)] = detect_image_batch(
                model=model,
                device=device,
                images=batch,
                criteria_iou=criteria_iou,
                max_output_iou=max_output_iou,
                prob_threshold=prob_threshold,
                use_head=use_head,
            )

            batch = []
            pbar.update(1)
            count += 1

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    return bbx_batch


def detect_videos_from_folder(
    model,
    device=Configs.device,
    path_to_data=Configs.path_data,
    batch_size=Configs.batch_size,
    criteria_iou=Configs.decode_result["criteria"],
    max_output_iou=Configs.decode_result["max_output"],
    prob_threshold=Configs.decode_result["pic_threshold"],
    use_head=Configs.use_head,
):
    print("run detect video")
    model.eval()
    if isinstance(path_to_data, str):
        path_to_data = Path(path_to_data)

    mp4 = list(path_to_data.rglob("*.mp4"))
    avi = list(path_to_data.rglob("*.avi"))

    videos = []
    videos.extend(mp4)
    videos.extend(avi)

    if len(videos) == 0:
        return "no video found"
    ans = {}
    ans["batch_size"] = batch_size

    for video in videos:
        print(video)
        ans[str(video)] = {}

        orginal_name = video.name
        orginal_name = orginal_name.rsplit(".", 1)[0]

        ans[str(video)] = detect_one_video(
            model=model,
            video=str(video),
            device=device,
            batch_size=batch_size,
            criteria_iou=criteria_iou,
            max_output_iou=max_output_iou,
            prob_threshold=prob_threshold,
            use_head=use_head,
        )

    return ans

import os
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from detection.config.config import Configs
from detection.detect_images import (detect_image_batch,
                                     detect_images_from_folder)
from detection.detect_video import detect_one_video, detect_videos_from_folder
from detection.ssd.create_model import nvidia_ssd
from detection.ssd.Detection_model import Detection_model

model = Detection_model()

app = FastAPI(
    title="Detection people with ssd300",
    description="Using ssd300 for detect people",
)


@app.on_event("startup")
async def startup_event():
    print("Server started ")


def load_image(data):
    return Image.open(BytesIO(data))


@app.post("/detect_image/")
async def detect_image(
    file: UploadFile = File(...),
    criteria_iou: Annotated[
        float, Path(title="riteria_iou", gt=0, le=1)
    ] = Configs.decode_result["criteria"],
    max_output_iou: Annotated[
        int, Path(title="max_output_iou", gt=0)
    ] = Configs.decode_result["max_output"],
    prob_threshold: Annotated[
        float, Path(title="prob_threshold", ge=0, le=1)
    ] = Configs.decode_result["pic_threshold"],
):
    image = load_image(await file.read())
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    return detect_image_batch(
        model=model.model,
        images=[img],
        criteria_iou=criteria_iou,
        max_output_iou=max_output_iou,
        prob_threshold=prob_threshold,
        use_head=model.use_head,
    )["img_num_0"]


@app.post("/detect_video/")
async def detect_video(
    file: UploadFile = File(...),
    batch_size: Annotated[
        int, Path(title="batch_size", gt=0)
    ] = Configs.batch_size,
    criteria_iou: Annotated[
        float, Path(title="riteria_iou", gt=0, le=1)
    ] = Configs.decode_result["criteria"],
    max_output_iou: Annotated[
        int, Path(title="max_output_iou", gt=0)
    ] = Configs.decode_result["max_output"],
    prob_threshold: Annotated[
        float, Path(title="prob_threshold", ge=0, le=1)
    ] = Configs.decode_result["pic_threshold"],
):
    # https://www.reactfix.com/2022/12/fixed-how-to-upload-list-of-videos.html
    temp = NamedTemporaryFile(delete=False)
    print("f", file.filename)
    extension = file.filename.split(".")[-1] in ("mp4", "avi")
    if not extension:
        raise HTTPException(
            status_code=403, detail="Video must be in mp4 or avi format!"
        )
    try:
        try:
            contents = file.file.read()
            with temp as f:
                f.write(contents)
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()
        bbx = detect_one_video(
            model=model.model,
            video=temp.name,
            device=model.device,
            batch_size=batch_size,
            criteria_iou=criteria_iou,
            max_output_iou=max_output_iou,
            prob_threshold=prob_threshold,
        )
    except Exception:
        return {"message": "There was an error processing the file"}

    finally:
        os.remove(temp.name)

    return {"batch_size": batch_size, "bbx": bbx}


@app.post("/detect_images_from_folder/{path_to_data}")
async def detect_image_from_folder(
    path_to_data: str,
    criteria_iou: Annotated[
        float, Path(title="riteria_iou", gt=0, le=1)
    ] = Configs.decode_result["criteria"],
    max_output_iou: Annotated[
        int, Path(title="max_output_iou", gt=0)
    ] = Configs.decode_result["max_output"],
    prob_threshold: Annotated[
        float, Path(title="prob_threshold", ge=0, le=1)
    ] = Configs.decode_result["pic_threshold"],
):
    ans = detect_images_from_folder(
        model=model.model,
        device=model.device,
        path_to_data=path_to_data,
        criteria_iou=criteria_iou,
        max_output_iou=max_output_iou,
        prob_threshold=prob_threshold,
        use_head=model.use_head,
    )

    return ans


@app.post("/detect_video_from_folder/{path_to_data}")
async def detect_video_from_folder(
    path_to_data: str,
    batch_size: Annotated[
        int, Path(title="batch_size", gt=0)
    ] = Configs.batch_size,
    criteria_iou: Annotated[
        float, Path(title="riteria_iou", gt=0, le=1)
    ] = Configs.decode_result["criteria"],
    max_output_iou: Annotated[
        int, Path(title="max_output_iou", gt=0)
    ] = Configs.decode_result["max_output"],
    prob_threshold: Annotated[
        float, Path(title="prob_threshold", ge=0, le=1)
    ] = Configs.decode_result["pic_threshold"],
):
    ans = detect_videos_from_folder(
        model=model.model,
        device=model.device,
        batch_size=batch_size,
        path_to_data=path_to_data,
        criteria_iou=criteria_iou,
        max_output_iou=max_output_iou,
        prob_threshold=prob_threshold,
        use_head=model.use_head,
    )

    return ans


@app.put("/load_weigth_for_model/")
def load_weigth_for_model(
    label_num: Annotated[int, Path(title="batch_size", gt=0)] = 3,
    device: Literal["cpu", "cuda"] = Configs.device,
    file: UploadFile = File(...),
):
    temp = NamedTemporaryFile(delete=False)
    try:
        contents = file.file.read()
        with temp as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    model.model = nvidia_ssd(
        pretrained_default=False,
        pretrainded_custom=True,
        path=temp.name,
        device=device,
        label_num=label_num,
    )
    model.label_num = label_num
    model.device = device
    model.weight = file.filename
    model.pretrainded_custom = True
    model.pretrained_default = False

    return {"status": "set pretrained model"}


@app.put("/model_setup/")
def setup_model(
    device: Literal["cpu", "cuda"] = Configs.device,
    use_head: bool = False,
):
    if device == "cuda" and torch.cuda.is_available():
        model.model.cuda()
        model.device = "cuda"
    else:
        device = "cpu"

    if device == "cpu":
        model.model.to("cpu")
        model.device = "cpu"

    if use_head and model.pretrainded_custom is True:
        model.use_head = True

    return {
        "device": model.device,
        "use_head": model.use_head,
        "model_pretrained_default": model.pretrained_default,
    }


@app.get("/model/info")
def get_model_info():
    return {
        "pretrained_default": model.pretrained_default,
        "pretrainded_custom": model.pretrainded_custom,
        "device ": model.device,
        "label_num": model.label_num,
        "weight": model.weight,
    }


def process_video(filename):
    print("Processing " + filename)
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", gray)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


@app.post("/detect_video_1/")
async def detect_video_1(file: UploadFile = File(...)):
    # https://www.reactfix.com/2022/12/fixed-how-to-upload-list-of-videos.html
    temp = NamedTemporaryFile(delete=False)
    print("f", file.filename)
    extension = file.filename.split(".")[-1] in ("mp4", "avi")
    if not extension:
        raise HTTPException(
            status_code=403, detail="Video must be in mp4 or avi format!"
        )
    try:
        contents = file.file.read()
        with temp as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    process_video(temp.name)
    os.remove(temp.name)

    return 0


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

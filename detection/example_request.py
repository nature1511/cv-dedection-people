import requests

from detection.utils.utils import (draw_bboxes_and_save_image,
                                   draw_boxes_and_save_video)

BASE_URL = "http://127.0.0.1:8000"

weight = "weight\state best_model_at_adam1.pth"
# "data\\2.jpg"
api_host = "http://127.0.0.1:8000/"


weigth_w = {"file": open(weight, "rb")}
params1 = {
    "criteria_iou": "0.5",
    "max_output_iou": "200",
    "prob_threshold": "0.3",
}

type_rq1 = "load_weigth_for_model"

weigth_w = {"file": open(weight, "rb")}
response1 = requests.put(api_host + type_rq1, files=weigth_w, params=params1)
print(response1.json())


type_rq2 = "detect_image"
input_image = "data\\1.jpg"
params2 = {
    "criteria_iou": "0.5",
    "max_output_iou": "200",
    "prob_threshold": "0.3",
}
files = {"file": open(input_image, "rb")}
response2 = requests.post(api_host + type_rq2, files=files, params=params2)
print(response2.json())

d = {}
d[input_image] = response2.json()

draw_bboxes_and_save_image(
    detect_res=d,
    path_new_data="",
    save_image=False,
    show_image=True,
)


type_rq3 = "detect_video"
input_video_name = "data\\video_1.mp4"
params3 = {
    "batch_size": "64",
    "criteria_iou": "0.5",
    "max_output_iou": "200",
    "prob_threshold": "0.3",
}
files = {"file": open(input_video_name, "rb")}
response3 = requests.post(api_host + type_rq3, files=files, params=params3)
print(response3.json().keys())
d3 = {}
d3["batch_size"] = response3.json()["batch_size"]
d3[input_video_name] = response3.json()["bbx"]


draw_boxes_and_save_video(d3, use_head=False)
# print(response2)

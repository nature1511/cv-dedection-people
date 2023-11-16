import json
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class CrowdHuman(Dataset):
    def __init__(self, img_folder, annotate_file, transform=None, use_head_bbx=True):
        super().__init__()
        self.img_folder = img_folder
        self.annotate_file = annotate_file
        self.transform = transform
        self.use_head_bbx = use_head_bbx

        print("load annotation file: ", self.annotate_file)
        with open(self.annotate_file, "r") as f:
            dataset = json.load(f)

        self.imgs = dict()
        for img in dataset["images"]:
            self.imgs[img["id"]] = img

        self.label_map = {}
        self.label_info = {}

        # 0 stand for the background
        self.label_info[0] = "background"
        self.label_info[1] = "person"

        self.label_map["background"] = 0
        self.label_map["person"] = 1

        if self.use_head_bbx:
            self.label_info[2] = "head"
            self.label_map["head"] = 2

        self.imgs_with_anns = defaultdict(list)
        for ann in dataset["annotations"]:
            self.imgs_with_anns[ann["image_id"]].append(ann)

        self.ids = list(sorted(self.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        anno = self.imgs_with_anns[img_id]
        file_name = self.imgs[img_id]["file_name"]
        img = Image.open(self.img_folder + file_name).convert("RGB")

        wtot, htot = img.size
        boxes = []
        target = []
        for obj in anno:
            l, t, w, h = obj["vbox"]
            r, b = l + w, t + h
            target.append(1)
            boxes.append([l / wtot, t / htot, r / wtot, b / htot])
        if self.use_head_bbx:
            for obj in anno:
                l, t, w, h = obj["hbox"]
                r, b = l + w, t + h
                target.append(2)
                boxes.append([l / wtot, t / htot, r / wtot, b / htot])


        bbox_sizes = torch.tensor(boxes)
        bbox_labels = torch.tensor(target)
        if self.transform is not None:
            img, (htot, wtot), bbox_sizes, bbox_labels = \
                self.transform(img, (htot, wtot), bbox_sizes, bbox_labels)

        else:
            pass

        return img, img_id, (htot, wtot), bbox_sizes, bbox_labels  # l t r b


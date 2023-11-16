import json
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


class Crowdhuman2coco:
    def __init__(
        self,
        path_to_annotation,
        path_to_data,
        path_to_new_data,
        path_to_json,
        im_start=0,
        number_images=100,
    ):
        if isinstance(path_to_annotation, str):
            path_to_annotation = Path(path_to_annotation)
        self.path_to_annotation = path_to_annotation

        if isinstance(path_to_data, str):
            path_to_data = Path(path_to_data)
        self.path_to_data = path_to_data

        if isinstance(path_to_new_data, str):
            path_to_new_data = Path(path_to_new_data)
        self.path_to_new_data = path_to_new_data

        if isinstance(path_to_json, str):
            path_to_json = Path(path_to_json)
        self.path_to_json = path_to_json

        self.number_images = number_images
        self.im_start = im_start

        path_to_new_data.mkdir(parents=True, exist_ok=True)

    def readlines(self):
        print("start read odgt file ")
        with open(self.path_to_annotation, "r") as f:
            lines = f.readlines()

        name = self.path_to_annotation.name.split(".")[0]
        print(f"{len(lines)} images in CrowdHuman {name} dataset")
        return [json.loads(line.strip("\n")) for line in lines]

    def convert2coco(
        self, args_head=1, args_rm_hocc=0, args_rm_hunsure=0, args_rm_hignore=0
    ):
        records = self.readlines()
        json_dict = {
            "images": [],
            "annotations": [],
            # "annotations1": [],
            "categories": [],
        }  # coco format

        bbox_id = 1
        categories = {}
        print("start convert")
        pbar = tqdm(total=self.number_images)
        count_save = 0
        for i, image_dict in enumerate(records):
            if i < self.im_start:
                #print(i)
                continue
            file_name = image_dict["ID"] + ".jpg"
            img = Image.open(self.path_to_data / file_name)

            if img.mode != "RGB":
                img.show()
                continue

            image = {
                "file_name": file_name,
                "height": img.size[1],
                "width": img.size[0],
                "id": image_dict["ID"],
            }
            gt_box = image_dict["gtboxes"]  # A list contains dicts
            annotation_mas = []
            for _, instance in enumerate(gt_box):
                annotation = {}
                category = instance["tag"]

                if category not in categories:
                    new_id = len(categories) + 1
                    categories[category] = new_id

                annotation["category_id"] = categories[category]

                if category not in categories:
                    new_id = len(categories) + 1
                    categories[category] = new_id
                annotation["vbox"] = instance["vbox"]

                if args_head:
                    attr = instance["head_attr"]
                    if args_rm_hocc:
                        if attr.get("occ", 1) == 1:
                            continue
                    if args_rm_hunsure:
                        if attr.get("unsure", 1) == 1:
                            continue
                    if args_rm_hignore:
                        if attr.get("ignore", 1) == 1:
                            continue
                    annotation["hbox"] = instance["hbox"]

                annotation["image_id"] = image_dict["ID"]
                annotation["id"] = bbox_id
                bbox_id += 1
                annotation_mas.append(annotation)
                # json_dict['annotations'].append(annotation)

            if len(annotation_mas) == 0:  # skip image without annotation
                continue

            json_dict["images"].append(image)
            json_dict["annotations"].extend(annotation_mas)
            img.save(self.path_to_new_data / file_name)

            pbar.update(1)
            count_save += 1
            if count_save == self.number_images:
                break

        for cate, cid in categories.items():
            cat = {"supercategory": cate, "id": cid, "name": cate}
            json_dict["categories"].append(cat)
        pbar.close()
        json_path = self.path_to_json
        print("start write json")
        json_fp = open(json_path, "w")
        json_str = json.dumps(json_dict)
        json_fp.write(json_str)
        json_fp.close()
        print(f"Json file have been dumped to {json_path}")

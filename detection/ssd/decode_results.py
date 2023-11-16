import numpy as np

from detection.config.config import Configs
from detection.ssd.utils_ssd300 import Encoder, dboxes300_coco


class Processing:
    @staticmethod
    def decode_results(
        predictions,
        criteria=Configs.decode_result["criteria"],
        max_output=Configs.decode_result["max_output"],
    ):
        dboxes = dboxes300_coco()
        encoder = Encoder(dboxes)
        ploc, plabel = [val.float() for val in predictions]
        results = encoder.decode_batch(
            ploc, plabel, criteria=criteria, max_output=max_output
        )
        return [
            [pred.detach().cpu().numpy() for pred in detections]
            for detections in results
        ]

    @staticmethod
    def pick_best(detections, threshold=Configs.decode_result["pic_threshold"]):
        _, _, confidences = detections
        best = np.argwhere(confidences > threshold)[:, 0]
        return [pred[best] for pred in detections]

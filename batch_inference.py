import cv2
from unilm.dit.object_detection.ditod import add_vit_config
import torch

from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

import time
import os
from PIL import Image
import numpy as np

# Step 1: instantiate config
cfg = get_cfg()
add_vit_config(cfg)
cfg.merge_from_file("maskrcnn_dit_base.yaml")

# Step 2: add model weights to config
cfg.MODEL.WEIGHTS = "./ckpts/model_final.pth"

# Step 3: set device
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Step 4: Set threshold

# balance obtaining high recall with not having too many low precision
# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# detections that will slow down inference post processing steps (like NMS)
# A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
# inference.
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1

# set number of classes: only 1 class in this case
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# Step 5: define model
predictor = DefaultPredictor(cfg)

# image in numpy
def analyze_image(img):
    start_time = time.perf_counter()

    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    md.set(thing_classes=["blob"]) ## added BG (background) in addition of custom class https://github.com/matterport/Mask_RCNN/issues/982
    
    output = predictor(img)["instances"]
    v = Visualizer(img[:, :, ::-1],
                    md,
                    scale=1.0,
                    instance_mode=ColorMode.IMAGE_BW)
    ## ColorMode.IMAGE SEGMENTATION IMAGE_BW
    ## https://detectron2.readthedocs.io/en/latest/_modules/detectron2/utils/visualizer.html
    result = v.draw_instance_predictions(output.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]

    run_time = time.perf_counter() - start_time
    return run_time, result_image



def main_fn():
    img_dir = "./PubLayNet_data_sample"
    out_dir = "../data/inference_output/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = os.listdir(img_dir)
    for i in files:
        if i.endswith(("jpg", ".png")): ### validation set are exclusively jpg or png format
            image_path = os.path.join(img_dir, i)
            out_path = os.path.join(out_dir, i)
            img = np.asarray(Image.open(image_path))
            run_time, output_arr = analyze_image(img)
            
            output_img = Image.fromarray(output_arr)
            output_img.save(out_path)

            print(f'model: publaynet_dit-b_cascade finetuned')
            print(f'{run_time} seconds')
            print("output saved as:", out_path)
    print('All Done!')

if __name__ == '__main__':
    main_fn()

# reference: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

import torch
import detectron2
from unilm.dit.object_detection.ditod import add_vit_config

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from PIL import Image

# import common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# ignore python warning (optional)
import os
import warnings
warnings.filterwarnings("ignore")


# register validation dataset
register_coco_instances("my_val", {}, "/home/ec2-user/data/test_coco.json", "/home/ec2-user/data/test/")

# 1 instantiate config
cfg = get_cfg()
add_vit_config(cfg)
cfg.merge_from_file("./cascade_dit_base.yaml")

# 2 add model weights config
cfg.MODEL.WEIGHTS = "/home/ec2-user/model/dit-base-cascade-tuned2-318k.pth"

# 3: set device
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
# A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
# inference.
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1

# only 1 class
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# define model
predictor = DefaultPredictor(cfg)

# evaluation for the given data
evaluator = COCOEvaluator("my_val", cfg, False, output_dir="../data/test_cascade_tuned2_318k_res/")
val_loader = build_detection_test_loader(cfg, "my_val")
inference_on_dataset(predictor.model, val_loader, evaluator)

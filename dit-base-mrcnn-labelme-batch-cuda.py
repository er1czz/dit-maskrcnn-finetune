'''
Update with class mapping in case of multi-class output
2024/01/04 Eric

Code that generates labelme json files by CUDA
Input: img directory
2023/10/27 Eric

Performance Note:
AWS g4dn.2xlarge (8 vCPU, 1 CUDA with 16 GB VRAM)
benchmark test: 16 images
    g4dn.2xlarge Pool(2): 177.89 seconds 20 - 25 s per img
    g4dn.2xlarge Pool(4): 210.52 seconds 40 - 60 s per img
    g4dn.2xlarge Pool(8): 361.72 seconds 160 - 200 s per img
However, 
    g4dn.2xlarge CUDA   :  11.16 second 0.6 - 0.8 s per img
'''

import torch
import detectron2
from unilm.dit.object_detection.ditod import add_vit_config

# Some basic setup:

# import some common libraries
import numpy as np
import os, json, cv2, time
#import multiprocessing

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# 1 instantiate config
cfg = get_cfg()
add_vit_config(cfg)
cfg.merge_from_file("./maskrcnn_dit_base.yaml")

# 2 add model weights config
cfg.MODEL.WEIGHTS = "./dit-base-mrcnn-tuned.pth"

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

# 4: define model
predictor = DefaultPredictor(cfg);

# 5: class mapping
cls_map = {0:'blob'}

def json_gen(img_path, json_path, pred_boxes, pred_classes, cls_map): 
# generate labelme label json
    image=cv2.imread(img_path) 
    g={} 
    g["version"]="5.1.1"
    g["flags"]={} 
    g["shapes"]=[] 
    for i in range(len(pred_boxes)):
        box = pred_boxes[i] 
        cls = pred_classes[i]
        g_int={}
        g_int["label"]= cls_map[cls]
        x1,y1,x2,y2=int(box[0]), int(box[1]), int(box[2]), int(box[3]) 
        g_int["points"]=[[int(x1),int(y1)],[int(x2),int(y2)]] 
        g_int["group_id"]=None 
        g_int["shape_type"]="rectangle" 
        g_int["flags"]={} 
        g["shapes"].append(g_int) 
    
    g["imagePath"]=img_path.split("/")[-1] 
    g["imageData"]=None
    g["imageHeight"]=image.shape[0]
    g["imageWidth"]=image.shape[1] 

    with open(json_path, 'w') as json_file: 
        json.dump(g, json_file,indent=4) 
    return g


def main_fun(img_path):
    # from input image to labelme json

    start_time = time.time()
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output = predictor(image)
    res = output['instances'].to("cpu")
    pred_boxes = res.pred_boxes.tensor.numpy()
    pred_classes = res.pred_classes.numpy()
    json_path = os.path.splitext(img_path)[0]+'.json'
    json_gen(img_path, json_path, pred_boxes, pred_classes, cls_map) 
    print(f'processed: {img_path} with output: {json_path}')
    print(f'{round((time.time() - start_time), 2)}s")\n')


if __name__ == "__main__":
    start_time0 = time.time()
    
    print(f"model is ready: {round((time.time() - start_time0), 2)}s")

    dir_path =  input("enter input image directory, e.g. 'test/': ")
    if dir_path[-1] != '/':
        dir_path += '/'
    print('>>> Thank you! Model is now working. No more input please. <<<')

    start_time = time.perf_counter()
    files = os.listdir(dir_path)

    for i in files:
        image_path = os.path.join(dir_path, i)
        main_fun(image_path)
        
    finish_time = time.perf_counter()
    print(f" >>> All tasks have been completed. Total inference time {finish_time-start_time} seconds. Thanks for using this tool! <<<")

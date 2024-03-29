# Document Layout Analysis from Data Labeling to Model Retraining
- keyword: ```computer vision```, ```object detection```,  ```document layout analysis```,  ```Vision Transformer (ViT)```, ```Document image Transformer (DiT)```, ```fine-tune```, ```retrain```, ```PyTorch```, ```detectron2```
- highlight:
  - ```label conversion from labelme bboxes to coco masks```
  - ```retrain model pretrained for multi-classes object detection into single-class```

<p align="center"><img src="https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/example_coco_masks.png" style = "border:10px solid white"> <img src="https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/example_labelme_bbox.png" style = "border:10px solid white"></p> 
<p align="center">LEFT inset: COCO mask labels for multi-classes detection</p>
<p align="center">RIGHT inset: labelme rectangle mask labels (converted from bounding boxes) for single-class detection</p>

## 1 Custom data labeling and preprocess
### Note1: common datasets for document layout
- DocLayNet (2023) 28GB, 80863 pages {financial_reports,scientific_articles,laws_and_regulations, government_tenders, manuals, patents}
- DocBank (2020) 50GB, 500K pages from arXiv.com {scientifi articles}
- PubLayNet (2019) 96GB, over 1 million pdf from PubMed {scientific articles}
- RVL-CDIP (2015) 37GB, 400,000 grayscale images, 16 classes
  - {"letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication", "specification", "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo"}
- PRImA Layout Analysis Dataset (2009) 1240 pages {magazines pages, technical articles}
- *somewhate relevant* HierText (2023)  11639 images {hierarchical annotations of text in natural scenes and documents}
  
### Note2: labeling tools
- labelme  https://github.com/wkentaro/labelme   
  - very basic functions and app can crush sometimes (remember to enable autosave)
  - outut label json file for each image separately 
  - can use third party tools to convert labelme output to cooc format: such as https://github.com/fcakyon/labelme2coco
- label studio https://labelstud.io/
  - mature but with decent learning curve
  - can natively output label in coco format (therefore, label data of all the images are compiled into one json file)

### LabelMe label output format
- example json file below shows three different labels in both rectangle (bbox) and polygon (mask) formats
- point coordinates for rectangle ```[[x1, y1], [x3, y3]]```
- point coordinates for polygon (mask) ```[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]```
- use this custom script to convert labelme to coco (all-in-one) [master_convert.py](https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/master_convert.py)
```
{"imagePath": "123.jpg",
  "imageData": null,
  "shapes":
      [
      {"shape_type": "rectangle", "points": [[204.9180327868852, 131.14754098360655], [1529.5081967213114, 1944.2622950819673]], "flags": {}, "group_id": null, "label": "section"},
      {"shape_type": "rectangle", "points": [[1585.245901639344, 78.68852459016394], [200.97022415523588, 1.4210854715202004e-14]], "flags": {}, "group_id": null, "label": "Header"},
      {"shape_type": "rectangle", "points": [[204.9180327868852, 1957.377049180328], [537.7049180327868, 2054.0983606557375]], "flags": {}, "group_id": null, "label": "section"},
      {"shape_type": "polygon", "points": [[204.9180327868852, 131.14754098360655], [204.9180327868852, 1944.2622950819673], [1529.5081967213114, 1944.2622950819673], [1529.5081967213114, 131.14754098360655]], "flags": {}, "group_id": null, "label": "section"},
      {"shape_type": "polygon", "points": [[1585.245901639344, 78.68852459016394], [1585.245901639344, 1.4210854715202004e-14], [200.97022415523588, 1.4210854715202004e-14], [200.97022415523588, 78.68852459016394]], "flags": {}, "group_id": null, "label": "Header"},
      {"shape_type": "polygon", "points": [[204.9180327868852, 1957.377049180328], [204.9180327868852, 2054.0983606557375], [537.7049180327868, 2054.0983606557375], [537.7049180327868, 1957.377049180328]], "flags": {}, "group_id": null, "label": "section"}
      ],
  "version": "5.1.1",
  "flags": {},
  "imageHeight": 2200,
  "imageWidth": 1700}
```

## 2 Setup environment
- below is the version of PyTorch and detectron2 I installed and ran successfully on both AWS EC2 g4dn.2xlarge instance (NVIDIA Turing T4 16GB VRAM) and a local workstation (NVIDIA Turing TU104 8GB VRAM)
```
torch:  2.0 ; cuda:  cu118
detectron2: 0.6
```
- Install cuda and then PyTorch
- Install [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (Building detectron2 from source is recommended)
- Then ```pip install -r dit_requirement.txt``` [dit_requirement.txt](https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/dit_requirement.txt)
- Clone unilm repo (DiT is part of unilm) ```git clone https://github.com/microsoft/unilm.git```
  - If you use Python3.10 or above, you will encounter this error ```ImportError: cannot import name 'Iterable' from 'collections'```
  - Please modify *unilm/dit/object_detection/ditod/table_evaluation/data_structure.py* replace line 6 as ```Iterable from collections.abc```
  - for more details, please check **DiT_tutorial.ipynb**

## 3 Model retraining
- please refer to [DiT_tutorial.ipynb](https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/DiT_tutorial.ipynb)
- warmup learning rate
```
# train specs
cfg.SOLVER.IMS_PER_BATCH = 2  # batch size
cfg.SOLVER.MAX_ITER = 100000   # max_iteration(100000) = images_count(500) * epoch(400) / batch_size (2)
cfg.SOLVER.BASE_LR = 25e-5  # LR (0.00025)
cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
cfg.SOLVER.WARMUP_ITERS = int(0.2*cfg.SOLVER.MAX_ITER)
cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
cfg.SOLVER.AMP.ENABLED = True
```
- data augmentation during training (!!! **issue** could be conflict in the source code)
  - error message: *AttributeError: Cannot find field 'gt_masks' in the given Instances!*
- [DiT trainer source code](https://github.com/microsoft/unilm/blob/master/dit/object_detection/ditod/mytrainer.py)
- [Detectron2 data augmentation source code](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/transforms/augmentation_impl.py)
```
# data augmentation during retrain
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
class MyTrainer_Aug(MyTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True,
                                use_instance_mask = True, # must enable, otherwise gt_mask error
                                augmentations=[T.AugmentationList([
                                   T.RandomBrightness(0.9, 1.1),
                                   T.RandomFlip(prob=0.5),
                                   #T.Resize((800, 1200)) # h, w 
                                   T.ResizeShortestEdge(800, 1200) # shorter edge will be scaled to 800 and longer edge will be no larger than 2000 (ratio locked) OUT OF VRAM
                                   ])]
                                )
        return build_detection_train_loader(cfg, mapper=mapper)
```
  - use ```trainer = MyTrainer_Aug(cfg)``` instead of ```trainer = MyTrainer(cfg)```
## 4 Evaluation (performance metrics)
- [python script](https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/evaluation.py)

## 5. Batch inference (active learning)
- Script that can use retrained model to generate output in labelme format
  - Inference by CPU (even with multiprocessing) is much slower (two orders of magnitude) than by CUDA of single process.
  - e.g. AWS g4dn.2xlarge CUDA inference of 500 images took about 340 s.
- [CUDA version](https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/dit-base-mrcnn-labelme-batch-cuda.py) (~0.6 s per image)
- [CPU version, multiprocessing](https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/dit-base-mrcnn-labelme-batch-cuda.py) (+20 s per image)

## 6. Data Augmentation (create a new dataset)
- [Albumentations](https://albumentations.ai/), a Python Library for image augmentation
- [A custom interactive script](https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/img_bbox_aug_gen.py) that will create a new dataset with ' _AuG' inserted in the file and path names
  
## 7. Result
- combine image sets side-by-side by opencv hconcat
  -  [ 1 × 2 script](https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/img_nx2.py)
  -  [ 1 × 3 script](https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/img_nx3.py)
- please note the fine-tuning was carried out with a small dataset of 20 images.
- therefore, the inference result from this "lightly" tuned model (right inset below) is not ideal albeit the prediction is for single-class.

<p align="center">
  <img src="https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/result_original_5classes.png" title="original model for multi-classes detection" style = "border:10px solid white" width="300"> 
  <img src="https://github.com/er1czz/dit-maskrcnn-finetune/blob/main/result_tuned_1class.png" title="fine-tuned model for single-class detection" style = "border:10px solid white" width="300">
</p> 
<p align="center">LEFT inset: original model for multi-classes detection</p>
<p align="center">RIGHT inset: fine-tuned model for single-class detection</p>


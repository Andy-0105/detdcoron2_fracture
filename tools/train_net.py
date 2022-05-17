#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
from detectron2.utils.visualizer import Visualizer

import logging
import os
from collections import OrderedDict
import torch
import cv2
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA



# 註冊資料集

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
import pycocotools
#宣告類別，儘量保持
CLASS_NAMES =["fracture"]
# 資料集路徑
DATASET_ROOT = './datasets/coco'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'train2017')
VAL_PATH = os.path.join(DATASET_ROOT, 'val2017')

TRAIN_JSON = os.path.join(ANN_ROOT, 'instances_train2017.json')
#VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
VAL_JSON = os.path.join(ANN_ROOT, 'instances_val2017.json')

# 宣告資料集的子集
PREDEFINED_SPLITS_DATASET = {
    "coco_my_train": (TRAIN_PATH, TRAIN_JSON),
    "coco_my_val": (VAL_PATH, VAL_JSON),
}
#===========以下有兩種註冊資料集的方法，本人直接用的第二個plain_register_dataset的方式 也可以用register_dataset的形式==================
#註冊資料集（這一步就是將自定義資料集註冊進Detectron2）

def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco")

#=============================
# 註冊資料集和元資料
def plain_register_dataset():
    #訓練集
    DatasetCatalog.register("coco_my_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_my_train").set(thing_classes=CLASS_NAMES,  # 可以選擇開啟，但是不能顯示中文，這裡需要注意，中文的話最好關閉
                                             evaluator_type='coco', # 指定評估方式
                                             json_file=TRAIN_JSON,
                                             image_root=TRAIN_PATH)

    #DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "coco_2017_val"))
    #驗證/測試集
    DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_my_val").set(thing_classes=CLASS_NAMES, # 可以選擇開啟，但是不能顯示中文，這裡需要注意，中文的話最好關閉
                                           evaluator_type='coco', # 指定評估方式
                                           json_file=VAL_JSON,
                                           image_root=VAL_PATH)
# 檢視資料集標註，視覺化檢查資料集標註是否正確，
#這個也可以自己寫指令碼判斷，其實就是判斷標註框是否超越影象邊界
#可選擇使用此方法
def checkout_dataset_annotation(name="coco_my_val"):
    #dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    dataset_dicts = load_coco_json(VAL_JSON, VAL_PATH)
    print(len(dataset_dicts))
    for i, d in enumerate(dataset_dicts, 0):
        #print(d)
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)

        #cv2.imshow('show', vis.get_image()[:, :, ::-1])
        cv2.imwrite('out/'+str(i) + '.jpg',vis.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
        # if i == 200:
        #     break
# checkout_dataset_annotation()


# python tools/train_net.py \
# --num-gpus 1 \
# --config-file configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0025






def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

# def setup(args):
#     """
#     Create configs and perform basic setups.
#     """
#     cfg = get_cfg()
#     args.config_file = "../configs/COCO-Detection/mask_rcnn_R_50_FPN_3x.yaml"
#     cfg.merge_from_file(args.config_file)   # 從config file 覆蓋配置
#     cfg.merge_from_list(args.opts)          # 從CLI引數 覆蓋配置
#
#     # 更改配置引數
#     cfg.DATASETS.TRAIN = ("coco_my_train",) # 訓練資料集名稱
#     cfg.DATASETS.TEST = ("coco_my_val",)
#     cfg.DATALOADER.NUM_WORKERS = 4  # 單執行緒
#
#     cfg.INPUT.CROP.ENABLED = True
#     cfg.INPUT.MAX_SIZE_TRAIN = 2688 # 訓練圖片輸入的最大尺寸
#     cfg.INPUT.MAX_SIZE_TEST = 2688 # 測試資料輸入的最大尺寸
#     cfg.INPUT.MIN_SIZE_TRAIN = (2688, 2688) # 訓練圖片輸入的最小尺寸，可以設定為多尺度訓練
#     cfg.INPUT.MIN_SIZE_TEST = 640
#     #cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING，其存在兩種配置，分別為 choice 與 range ：
#     # range 讓影象的短邊從 512-768隨機選擇
#     #choice ： 把輸入影象轉化為指定的，有限的幾種圖片大小進行訓練，即短邊只能為 512或者768
#     cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'
#     #  本句一定要看下注釋！！！！！！！！
#     cfg.MODEL.RETINANET.NUM_CLASSES = 81  # 類別數+1（因為有background，也就是你的 cate id 從 1 開始，如果您的資料集Json下標從 0 開始，這個改為您對應的類別就行，不用再加背景類！！！！！）
#     #cfg.MODEL.WEIGHTS="/home/yourstorePath/.pth"
#     cfg.MODEL.WEIGHTS = "/root/xxx/model_final_5bd44e.pkl"    # 預訓練模型權重
#     cfg.SOLVER.IMS_PER_BATCH = 4  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size
#
#     # 根據訓練資料總數目以及batch_size，計算出每個epoch需要的迭代次數
#     #9000為你的訓練資料的總數目，可自定義
#     ITERS_IN_ONE_EPOCH = int(81 / cfg.SOLVER.IMS_PER_BATCH)
#
#     # 指定最大迭代次數
#     cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 50) - 1 # 12 epochs，
#     # 初始學習率
#     cfg.SOLVER.BASE_LR = 0.002
#     # 優化器動能
#     cfg.SOLVER.MOMENTUM = 0.9
#     #權重衰減
#     cfg.SOLVER.WEIGHT_DECAY = 0.0001
#     cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
#     # 學習率衰減倍數
#     cfg.SOLVER.GAMMA = 0.1
#     # 迭代到指定次數，學習率進行衰減
#     cfg.SOLVER.STEPS = (7000,)
#     # 在訓練之前，會做一個熱身運動，學習率慢慢增加初始學習率
#     cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
#     # 熱身迭代次數
#     cfg.SOLVER.WARMUP_ITERS = 1000
#
#     cfg.SOLVER.WARMUP_METHOD = "linear"
#     # 儲存模型檔案的命名資料減1
#     cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1
#
#     # 迭代到指定次數，進行一次評估
#     cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
#     #cfg.TEST.EVAL_PERIOD = 100
#
#     #cfg.merge_from_file(args.config_file)
#     #cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     default_setup(cfg, args)
#     return cfg
def setup(args):
    """
    Create configs and perform basic setups.
    """
    # epoch is MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ("coco_my_train",) # 訓練資料集名稱，修改
    cfg.DATASETS.TEST = ("coco_my_val",) # 訓練資料集名稱，修改
    cfg.MODEL.RETINANET.NUM_CLASSES = 1 # 修改自己的類別數
    cfg.SOLVER.MAX_ITER=270000
    cfg.freeze()
    # epoch is MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    plain_register_dataset()# 修改
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
#tensorboard --logdir=./output

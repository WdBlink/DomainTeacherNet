#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from datasets.pascal_voc import register_pascal_voc

from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer, DomainTeacherTrainer

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin
from detectron2.data import DatasetCatalog, MetadataCatalog

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
import os
from options.train_options import TrainOptions

os.environ["DETECTRON2_DATASETS"] = "/mnt/e/Dataset/"
print(os.getenv("DETECTRON2_DATASETS"))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    register_pascal_voc("destroyed_building_trainval_tif", "/mnt/e/Dataset/taining_data_2021-08-19", "trainval", year=2014,
                        class_names='1')
    register_pascal_voc("destroyed_building_test_tif", "/mnt/e/Dataset/taining_data_2021-08-19", "test", year=2014,
                        class_names='1')
    MetadataCatalog.get("destroyed_building_test_tif").set(thing_classes=["1"],  # 可以选择开启，但是不能显示中文，这里需要注意，中文的话最好关闭
                                           evaluator_type='coco')

    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
        trainer = Trainer(cfg)
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
        trainer = Trainer(cfg)
    elif cfg.SEMISUPNET.Trainer == 'DomainTeacherTrainer':
        Trainer = DomainTeacherTrainer
        trainer = Trainer(cfg, args)
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = TrainOptions().parse()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

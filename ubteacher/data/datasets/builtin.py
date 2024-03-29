# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import contextlib
from detectron2.data import DatasetCatalog, MetadataCatalog
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager
import io
import logging
from detectron2.data.datasets.cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from detectron2.data.datasets.cityscapes_panoptic import register_all_cityscapes_panoptic
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.structures import BoxMode
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
from pathlib import Path
import glob
import cv2

logger = logging.getLogger(__name__)
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

JSON_ANNOTATIONS_DIR = ""
_SPLITS_COCO_FORMAT = [
    ("destroy_201005069_unlable", "/test_tif/201005069"),
    ("destroy_0013_unlable", "/test_tif/0013")
]


def register_destroy_unlabel(root):
    for key, dataset_name in _SPLITS_COCO_FORMAT:
        meta = {}
        image_root = root + dataset_name
        register_destroy_unlabel_instances(key, meta, image_root)


def register_destroy_unlabel_instances(name, metadata, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(image_root, (str, os.PathLike)), image_root

    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_destroy_unlabel_file(image_root)
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        image_root=image_root, evaluator_type="destroy_voc", **metadata
    )


def load_destroy_unlabel_file(image_root):
    p = str(Path(image_root).absolute())
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    dataset_dicts = []
    for path in images:
        img_id = os.path.splitext(os.path.split(path)[1])[0]
        file_name = os.path.split(path)[1]
        img = cv2.imread(path)
        height, width = img.shape[:2]
        record = {}
        record["file_name"] = path
        record["height"] = height
        record["width"] = width
        record["image_id"] = img_id
        dataset_dicts.append(record)

    return dataset_dicts

# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_foggy_fine_{task}_train": ("cityscapes/leftImg8bit_foggy/train/", "cityscapes/gtFine/train/"),
    "cityscapes_foggy_fine_{task}_val": ("cityscapes/leftImg8bit_foggy/val/", "cityscapes/gtFine/val/"),
    "cityscapes_foggy_fine_{task}_test": ("cityscapes/leftImg8bit_foggy/test/", "cityscapes/gtFine/test/"),
}

def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )

# ==== Predefined splits for PASCAL VOC ===========\
CLASS_NAMES = ["1"]

def register_all_destroy_voc(root):
    SPLITS = [
        ("voc_destroy_trainval", "taining_data_2021-08-19", "trainval"),
        ("voc_destroy_train", "taining_data_2021-08-19", "train"),
        ("voc_destroy_val", "taining_data_2021-08-19", "val"),
        ("voc_destroy_test", "taining_data_2021-08-19", "test")
    ]
    for name, dirname, split in SPLITS:
        register_pascal_voc(name, os.path.join(root, dirname), split)
        MetadataCatalog.get(name).set(evaluator_type='destroy_voc')

def load_voc_instances(dirname: str, split: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".tif")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_pascal_voc(name, dirname, split, class_names=CLASS_NAMES):
    year = 2007 if "2007" in name else 2012
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, split=split, year=year
    )


os.environ["DETECTRON2_DATASETS"] = "/mnt/c/Dataset/"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_all_cityscapes(_root)
register_all_destroy_voc(_root)
register_destroy_unlabel(_root)


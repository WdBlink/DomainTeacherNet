import time

import cv2
import os
import torch
from torch.distributions.weibull import Weibull
from torch.distributions.transforms import AffineTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer, DomainTeacherTrainer

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin
import detectron2.data.transforms as T
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from tqdm import tqdm
from pathlib import Path
import glob
import numpy as np
import detectron2.data.detection_utils as utils
import PIL

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
def create_distribution(scale, shape, shift):
    wd = Weibull(scale=scale, concentration=shape)
    transforms = AffineTransform(loc=shift, scale=1.)
    weibull = TransformedDistribution(wd, transforms)
    return weibull


def compute_prob(x, distribution):
    eps_radius = 0.5
    num_eval_points = 100
    start_x = x - eps_radius
    end_x = x + eps_radius
    step = (end_x - start_x) / num_eval_points
    dx = torch.linspace(x - eps_radius, x + eps_radius, num_eval_points)
    pdf = distribution.log_prob(dx).exp()
    prob = torch.sum(pdf * step)
    return prob


def update_label_based_on_energy(logits, classes, unk_dist, known_dist):
    unknown_class_index = 80
    cls = classes
    lse = torch.logsumexp(logits[:, :5], dim=1)
    for i, energy in enumerate(lse):
        p_unk = compute_prob(energy, unk_dist)
        p_known = compute_prob(energy, known_dist)
        # print(str(p_unk) + '  --  ' + str(p_known))
        if torch.isnan(p_unk) or torch.isnan(p_known):
            continue
        if p_unk > p_known:
            cls[i] = unknown_class_index
    return cls


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


def crop_and_save(merge_img, crop_size, output_path, cfg):
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    elif cfg.SEMISUPNET.Trainer == 'DomainTeacherTrainer':
        Trainer = DomainTeacherTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if cfg.SEMISUPNET.Trainer == "ubteacher" or cfg.SEMISUPNET.Trainer == "DomainTeacherTrainer":
        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)
        ensem_ts_model.eval()
        DetectionCheckpointer(
            ensem_ts_model, save_dir=cfg.OUTPUT_DIR
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        predictor = ensem_ts_model.modelTeacher
    else:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        predictor = model

    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)


    shape = merge_img.shape[:2]
    ys = shape[0] // crop_size
    xs = shape[1] // crop_size
    if os.path.exists(output_path):
        pass
    else:
        os.makedirs(output_path)
    merge_img_gpu = torch.as_tensor(merge_img.astype('float32').transpose(2, 0, 1), device='cuda:0')
    with tqdm(total=ys*xs) as pbar:
        for y in range(ys):
            for x in range(xs):
                # cropped = merge_img[y*crop_size:(y+1)*crop_size, x*crop_size:(x+1)*crop_size, :]
                cropped = merge_img_gpu[:, y*crop_size:(y+1)*crop_size, x*crop_size:(x+1)*crop_size]
                path = os.path.join(output_path, f'{y}_{x}.tif')

                with torch.no_grad():
                    # height, width = cropped.shape[:2]
                    height, width = cropped.shape[1:]
                    # im = aug.get_transform(cropped).apply_image(cropped)
                    image = cropped
                    # image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))

                    inputs = {"image": image, "height": height, "width": width}
                    outputs = predictor([inputs])[0]

                save_img = True
                if save_img:
                    image_numpy = image.cpu().float().numpy()  # convert it into a numpy array
                    image_numpy = np.transpose(image_numpy, (1, 2, 0))
                    # imae_numpy = np.transpose(image_numpy, (1, 2, 0))[:, :, ::-1]  # post-processing: tranpose
                    v = Visualizer(image_numpy, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                    v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
                    img = v.get_image()

                    print(f'Save image to {path}')
                    try:
                        cv2.imwrite(path, img)
                    except Exception as err:
                        print(err)
                pbar.update(1)


def main(args):
    # Get image
    cfg = setup(args)

    # fnum = '1485'
    # file_name = '2_2'
    # root_path = '/mnt/e/datasets/test_tif'
    root_path = '/mnt/e/datasets/tmp'
    output_root = cfg.OUTPUT_DIR
    imgsz = 1000
    # file_path = os.path.join('/home/msi/Documents/Datasets/test_tif/Production_ORTH4_ortho_part_5_2',
    #                          file_name + '.tif')
    # original_image = cv2.imread(file_path)
    # model = '/home/fk1/workspace/OWOD/output/old/t1_20_class/model_0009999.pth'
    # model = '/home/fk1/workspace/OWOD/output/t1_THRESHOLD_AUTOLABEL_UNK/model_final.pth'
    # model = '/home/fk1/workspace/OWOD/output/t1_clustering_with_save/model_final.pth'
    # model = '/home/fk1/workspace/OWOD/output/t2_ft/model_final.pth'
    # model = '/home/fk1/workspace/OWOD/output/t3_ft/model_final.pth'
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.61
    # cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.8
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4

    # POSITIVE_FRACTION: 0.25
    # NMS_THRESH_TEST: 0.5
    # SCORE_THRESH_TEST: 0.05
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 21
    p = str(Path(root_path).absolute())  # os-agnostic absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

    for path in images:
        img_id = os.path.splitext(os.path.split(path)[1])[0]
        # img = cv2.imread(path)
        input_format = "BGR"
        PIL.Image.MAX_IMAGE_PIXELS = None
        img = utils.read_image(path, format=input_format)
        # input_format = cfg.INPUT.FORMAT
        #
        # if input_format == 'RGB':
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_path = os.path.join(output_root, img_id)
        crop_and_save(merge_img=img, crop_size=imgsz, output_path=output_path, cfg=cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    t0 = time.time()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    t1 = time.time()
    print(f'Total time is {t1-t0:.3f}s')

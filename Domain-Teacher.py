import torch
import os
import numpy as np
import cv2
import math
import sys
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
from xml.dom.minidom import parse
from utils.datasets import VOCDataset, DomainDataset
import utils.transforms as T
from utils.engine import train_one_epoch, evaluate
import utils.utils as utils
import torchvision
from models.detection.faster_rcnn import FastRCNNPredictor
from models.domainnet import DomainNet
import argparse
from torch.nn import MSELoss


def get_object_detection_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # replace the classifier with a new one, that has num_classes which is user-defined
    num_classes = 2  # 3 class (mark_type_1，mark_type_2) + background

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # 50%的概率水平翻转
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


def showbbox(model, img, result_path, device):
    # 输入的img是0-1范围的tensor
    model.eval()
    with torch.no_grad():
        '''
        prediction形如：
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        prediction = model([img.to(device)])

    print(prediction)

    img = img.permute(1, 2, 0)  # C,H,W → H,W,C，用来画图
    img = (img * 255).byte().data.cpu()  # * 255，float转0-255
    img = np.array(img).copy()  # tensor → ndarray

    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())

        label = prediction[0]['labels'][i].item()

        if label == 1:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
            cv2.putText(img, 'mark_type_1', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                        thickness=2)
        elif label == 2:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
            cv2.putText(img, 'mark_type_2', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        thickness=2)
    cv2.imwrite(result_path, img)


def main(opt):
    root = opt.root

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    # 3 classes, mark_type_1，mark_type_2，background
    num_classes = 2
    # use our dataset and defined transformations
    dataset = DomainDataset(root, get_transform(train=True))
    dataset_test = DomainDataset(root, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-100])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    domain_net = DomainNet(in_channels=6, out_channels=3, final_sigmoid=True)
    rcnn = torch.load(opt.weights)
    # rcnn = fasterrcnn_transformer_fpn(pretrained=False, progress=True, num_classes=num_classes,
    #                                    pretrained_backbone=False)
    # 或get_object_detection_model(num_classes)

    # move model to the right device
    domain_net.to(device)
    rcnn.to(device)

    # construct an optimizer
    params = [p for p in domain_net.parameters() if p.requires_grad]

    # SGD
    optimizer = torch.optim.SGD(params, lr=0.0003,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    # cos学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # let's train it for   epochs
    num_epochs = 50
    save_path = '/mnt/e/github/object-detection-zoo/runs/train/tmp/faster_rcnn.pkl'
    result_path = '/mnt/e/github/object-detection-zoo/runs/train/tmp/faster_rcnn.jpg'

    for epoch in range(num_epochs):
        # train for one epoch, printing every 50 iterations
        domain_net.train()
        # metric_logger = utils.MetricLogger(delimiter="  ")
        # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        # header = 'Epoch: [{}]'.format(epoch)

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        for images, targets, domain_images in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            domain_images = domain_net(images)
            loss_dict = rcnn(domain_images, targets)
            mseloss = torch.nn.MSELoss()
            losses = sum(loss for loss in loss_dict.values()) + mseloss(domain_images, images)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        # save model file
        print(f'save model to {save_path}')
        torch.save(model, save_path)
        print('==================================================')
        print('')

    # print("That's it!")
    # print('==================================================')
    print('====================begin test====================')
    model = torch.load(save_path)
    print(device)
    model.to(device)
    # evaluate(model, data_loader_test, device=device)
    dataset_test = VOCDataset(root, get_transform(train=False))
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])
    img, _ = dataset_test[0]
    showbbox(model, img, result_path=result_path, devive=device)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='E:\\github\\object-detection-zoo\\runs\\train\\faster_rcnn.pkl',
                        help='model.pt path(s)')
    parser.add_argument('--root', type=str, default='E:/Dataset/taining_data_2021-08-19',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1000, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-img', default=True, action='store_true', help='save img results')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect/Faster-Rcnn-mobilenetv3', help='save results to project/name')
    parser.add_argument('--name', default='Satellite', help='save results to project/name')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
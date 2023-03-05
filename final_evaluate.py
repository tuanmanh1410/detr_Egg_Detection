# Command line: python final_evaluate.py --coco_path --resume --num_classes --detail
import datetime
import argparse

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
from tta_engine import tta_evaluate
from models import build_model

# Library for evaluation bounding boxes
from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import torchvision.transforms as T
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
from matplotlib import pyplot as plt


# Library for read XML file
import xml.etree.ElementTree as ET


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--detail', default=False, type=bool, help='Show detail bounding boxes of each image')
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--num_classes', default=7, type=int,
                        help="Number of classes in dataset+1")
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

# Function to evaluate the model by computing the mAP score (COCO-API)
def Evaluate_AP(model, args):

    dataset_test = build_dataset(image_set='test', args=args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=2)

    # Evaluate
    base_ds = get_coco_api_from_dataset(dataset_test)
    model = model.to(args.device)
    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_test, base_ds, args.device, args.output_dir)

    return test_stats

# Preprocess image - Get list of images
def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images   

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_min, y_min, w, h = x.unbind(1)
    b = [x_min - 0.5 * w, y_min - 0.5 * h, x_min + 0.5 * w, y_min + 0.5 * h]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# Put everything on detect function
def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.8

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    #bboxes_scaled = bboxes_scaled.tolist()

    return probas[keep], bboxes_scaled

# Compute distance between two points (x1, y1) and (x2, y2)
def distance(x, y):
    return ((x[0] - y[0])**2 + (x[1] - y[1])**2)**0.5


def Get_GroundTruth_Bounding_Boxes(file_name):
    # Get information from the xml file
    tree = ET.parse(file_name)
    root = tree.getroot()
    # Read information from <object_count> tag
    object_count = int(root.find('object_count').text)
    Bbox_GT = []
    class_code_GT = []

    for member in root.findall('bndbox'):
        xmin = float(member.find('x_min').text)
        ymin = float(member.find('y_min').text)
        xmax = float(member.find('x_max').text)
        ymax = float(member.find('y_max').text)
        state = int(member.find('state').text)
        bndbox = [xmin, ymin, xmax, ymax]
        # Append bounding box to the Bbox_GT list
        Bbox_GT.append(bndbox)
        class_code_GT.append(state)
    
    return object_count, Bbox_GT, class_code_GT

def compute_overlap(boxes, query_boxes):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0])
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1])
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0]) *
                        (boxes[n, 3] - boxes[n, 1]) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def get_detections(bbox, scores, labels):

    """
     res = {label1: [[xmin, xmax, ymin, ymax, score], ...],
             label2: [[xmin, xmax, ymin, ymax, score], ...]]}
             labelC: [[xmin, xmax, ymin, ymax, score], ...]]}
    """

    res = dict()

    for i in range(len(bbox)):
        label = labels[i]
        if label not in res:
            res[label] = []
        box = bbox[i].detach().numpy().tolist() + [scores[i].item()]
        res[label].append(box)

    return res

def get_annotations(bbox, labels):
    """
     res = {label1: [[xmin, xmax, ymin, ymax], ...],
             label2: [[xmin, xmax, ymin, ymax], ...]]}
             labelC: [[xmin, xmax, ymin, ymax], ...]]}
    """

    res = dict()

    for i in range(len(bbox)):
        label = labels[i]
        if label not in res:
            res[label] = []
        box = bbox[i]
        res[label].append(box)

    return res

# Get the TP, FP, FN for each image following the IoU threshold
def evaluate_sample(bboxes_scaled, probas, object_count, Bbox_GT, class_code_GT, IOU_threshold, num_classes):
    # Initialize TP, FP, FN
    TP = {cl+1: [] for cl in range(num_classes)}
    FP = {cl+1: [] for cl in range(num_classes)}
    FN = {cl+1: 0 for cl in range(num_classes)}
    NUM_ANNOTATIONS = {cl+1: 0 for cl in range(num_classes)}
    SCORES = {cl+1: [] for cl in range(num_classes)}

    # Get the class code and its scores of the bounding box
    class_code, class_score = Get_ClassCode_with_score(probas)

    all_detections = get_detections(bboxes_scaled, class_score, class_code)
    all_annotations = get_annotations(Bbox_GT, class_code_GT)

    for label in range(1, num_classes+1):

        num_annotations = 0.0
        detections = []
        annotations = []

        if label in all_detections:
            detections = all_detections[label]
        
        if label in all_annotations:
            annotations = all_annotations[label]
        
        if len(detections) == 0 and len(annotations) == 0:
            continue
        
        NUM_ANNOTATIONS[label] = len(annotations)
        detected_annotations = []

        annotations = np.array(annotations, dtype=np.float64)
        for d in detections:
            SCORES[label].append(d[4])

            if len(annotations) == 0:
                FP[label].append(1)
                TP[label].append(0)
                continue
            
            overlaps = compute_overlap(np.expand_dims(np.array(d, dtype=np.float64), axis=0), annotations)
            assigned_annotation = np.argmax(overlaps, axis=1)
            max_overlap = overlaps[0, assigned_annotation]

            if max_overlap >= IOU_threshold and assigned_annotation not in detected_annotations:
                FP[label].append(0)
                TP[label].append(1)
                detected_annotations.append(assigned_annotation)
            else:
                FP[label].append(1)
                TP[label].append(0)
        
        FN[label] = NUM_ANNOTATIONS[label] - np.sum(TP[label])

    return TP, FP, FN, NUM_ANNOTATIONS, SCORES

            
def Get_ClassCode_with_score(scores):
    # scores is a tensor
    class_code = []
    class_scores = []
    for p in scores:
        cl = p.argmax()
        cl = cl.tolist()
        class_code.append(cl)
        class_scores.append(p[cl])
    return class_code, class_scores

def Get_ClassCode(scores):
    # scores is a tensor
    class_code = []
    for p in scores:
        cl = p.argmax()
        cl = cl.tolist()
        class_code.append(cl)
    return class_code

def compute_average_precision(TP, FP, SCORES, NUM_ANNOTATIONS):

    false_positives = np.array(FP)
    true_positives = np.array(TP)
    scores = np.array(SCORES)
    num_annotations = NUM_ANNOTATIONS

    # sort by score
    indices = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives = true_positives[indices]

    # compute false positive and true positives
    false_positives = np.cumsum(false_positives)
    true_positives = np.cumsum(true_positives)

    # compute recall and precision
    recall = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision = _compute_ap(recall, precision)

    return average_precision

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



if __name__ == '__main__':

    parser = argparse.ArgumentParser('EGG Evaluation starting ...', parents=[get_args_parser()])
    print("The evaluation code is based on : https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/eval.py")
    print("The evaluation code is based on : https://github.com/rbgirshick/py-faster-rcnn")

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Show the command line arguments
    print("Command line:")
    # Print the command line arguments
    print('python final_evaluate.py --resume ' + args.resume \
           + ' --batch_size ' + str(args.batch_size) \
           + ' --num_classes ' + str(args.num_classes) \
           + ' --coco_path ' + args.coco_path \
           + ' --detail ' + str(args.detail))
    print('------------------------------------------------------------------------------------------')

    print("Start time: ", datetime.datetime.now())

    transform = T.Compose([T.Resize(800),
                           T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], 
                                       [0.229, 0.224, 0.225])])

    model, criterion, postprocessors = build_model(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    DIR_TEST = os.path.join(args.coco_path, 'test')
    test_images = collect_all_images(DIR_TEST)
    NUM_CLASSES = args.num_classes - 1

    # Initialize Global TP, FP, FN
    Global_TP = {cl+1: [] for cl in range(args.num_classes)}
    Global_FP = {cl+1: [] for cl in range(args.num_classes)}
    NUM_ANNOTATIONS = {cl+1: 0 for cl in range(args.num_classes)}
    SCORES = {cl+1: [] for cl in range(args.num_classes)}
    total_num_tests = len(test_images)

    for n, image in enumerate(test_images):
        img = Image.open(image)
        # Replace the path of the image with the path of the xml file
        xml_file = image.replace('jpg', 'xml')
        # Get the number of ground truth bounding boxes and the ground truth bounding boxes
        object_count, Bbox_GT, class_code_GT = Get_GroundTruth_Bounding_Boxes(xml_file)

        print('===================================================================================================')
        print(f'[{n+1}/{total_num_tests}] Image: ', image)

        scores, boxes = detect(img, model, transform)
        TP, FP, FN, num_annotations, score = evaluate_sample(boxes, scores, object_count, Bbox_GT, class_code_GT, IOU_threshold = 0.5, num_classes=NUM_CLASSES)

        # Print TP, FP, FN in one line
        print(f'<Confusion Matrix for this Image>')
        print('|---------------|---------------|---------------|---------------|---------------|---------------|')
        print('|\tCLASS\t|\tTP\t|\tFP\t|\tFN\t|   Precision   |\tRecall\t|')
        print('|---------------|---------------|---------------|---------------|---------------|---------------|')
        for i in range(1, args.num_classes):
            TP_ = sum(TP[i])
            FP_ = sum(FP[i])
            FN_ = int(FN[i])
            if TP_ + FP_ == 0 and TP_ + FN_ == 0:
                print(f'|\t{i}\t|\t{TP_}\t|\t{FP_}\t|\t{FN_}\t|\t-\t|\t-\t|')
            elif TP_ + FP_ == 0:
                print(f'|\t{i}\t|\t{TP_}\t|\t{FP_}\t|\t{FN_}\t|\t-\t|\t{TP_/(TP_+FN_):.3f}\t|')
            elif TP_ + FN_ == 0:
                PRECISION = TP_/(TP_+FP_)
                print(f'|\t{i}\t|\t{TP_}\t|\t{FP_}\t|\t{FN_}\t|\t{PRECISION:.3f}\t|\t-\t|')
            else:
                PRECISION = TP_/(TP_+FP_)
                RECALL = TP_/(TP_+FN_) 
                print(f'|\t{i}\t|\t{TP_}\t|\t{FP_}\t|\t{FN_}\t|\t{PRECISION:.3f}\t|\t{RECALL:.3f}\t|')

            Global_TP[i] += TP[i]
            Global_FP[i] += FP[i]
            NUM_ANNOTATIONS[i] += num_annotations[i]
            SCORES[i] += score[i]
        print('|---------------|---------------|---------------|---------------|---------------|---------------|')

        classes = Get_ClassCode(scores)

        print('<Bounding box details>')
        print('Number of ground truth bounding boxes: ', object_count)
        print("The details of the ground truth bounding boxes: (Class Code + bounding boxes)")
        for i in range(len(Bbox_GT)):
            print(class_code_GT[i], Bbox_GT[i])

        print('Number of predicted bounding boxes: ', len(boxes)) 
        
        # Show the details of the predicted bounding boxes
        if (args.detail == True):
            print('Details of predicted bounding boxes: (Class Code + bounding boxes)')
            for i in range(len(boxes)):
                print(classes[i], boxes[i].detach().numpy().tolist())
        
        
    # Compute 

    # Print Global TP, FP, FN
    print('===================================================================================================')
    print(f'Global Confusion Matrix at threshold')
    print('|---------------|---------------|---------------|---------------|---------------|---------------|---------------|')
    print('|\tCLASS\t|\tTP\t|\tFP\t|\tFN\t|   Precision   |\tRecall\t|    F1-score   |')
    print('|---------------|---------------|---------------|---------------|---------------|---------------|---------------|')
    for i in range(1, args.num_classes):
        Global_TP_ = sum(Global_TP[i])
        Global_FP_ = sum(Global_FP[i])
        Global_FN_ = NUM_ANNOTATIONS[i] - Global_TP_
        if Global_TP_ + Global_FP_ != 0:
            PRECISION = Global_TP_/(Global_TP_+Global_FP_)
            RECALL = Global_TP_/NUM_ANNOTATIONS[i]
            F1_score = 2 * PRECISION * RECALL / (PRECISION + RECALL)
            print(f'|\t{i}\t|\t{Global_TP_}\t|\t{Global_FP_}\t|\t{Global_FN_}\t|\t{PRECISION:.3f}\t|\t{RECALL:.3f}\t|\t{F1_score:.3f}\t|')
        else:
            print(f'|\t{i}\t|\t{Global_TP_}\t|\t{Global_FP_}\t|\t{Global_FN_}\t|\t-\t|\t-\t|\t-\t|')
    print('|---------------|---------------|---------------|---------------|---------------|---------------|---------------|')

    print('Compute mAP....')
    mAP = 0
    for i in range(1, args.num_classes):
        ap = compute_average_precision(Global_TP[i], Global_FP[i], SCORES[i], NUM_ANNOTATIONS[i])
        mAP += ap
        print(f'Average Precision for class {i} : {ap:.3f}')

    mAP = mAP / NUM_CLASSES
    print(f"mAP@0.5 : {mAP:.3f}")
    print('===================================================================================================')
    print("Evaluation Complete: ", datetime.datetime.now())

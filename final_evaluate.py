# Command line: python final_evaluate.py --coco_path --resume --num_classes
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


# Library for read XML file
import xml.etree.ElementTree as ET


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
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

# Function to evaluate the model by computing the mAP score
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


def Evaluate_AP_EachClass(model, args):

    dataset_test = build_dataset(image_set='test', args=args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                    drop_last=False, collate_fn=utils.collate_fn, num_workers=2)


    # Evaluate
    base_ds = get_coco_api_from_dataset(dataset_test)

    model = model.to(args.device)

    # Hard code here
    CatIDs = [1,2,3,4,5,6,7]
    # End hard code

    G_mAP = []
    for CatID in CatIDs:
        test_stats, coco_evaluator = tta_evaluate(model, criterion, postprocessors, data_loader_test, base_ds, args.device, args.output_dir, CatID)
        G_mAP.append(test_stats["coco_eval_bbox"][1]) # Get the mAP50 of the current class

    return G_mAP

#**************Inference Phase***************#

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

def Select_Bounding_Boxes(probas, bboxes_scaled):

    # Finding the center coordinates of the bounding box
    center_x = (bboxes_scaled[:, 0] + bboxes_scaled[:, 2]) / 2
    center_y = (bboxes_scaled[:, 1] + bboxes_scaled[:, 3]) / 2

    # Combine center_x and center_y to get the center coordinates
    center = torch.stack((center_x, center_y), dim=1)
    center = center.tolist()

    # Remove the center coordinates that are too close to each other
    dist = 0
    index_remove = []
    for i in range(len(center)):
        for j in range(i + 1, len(center)):
            dist = distance(center[i], center[j])
            if dist < 50:
                # Remove the center with the lower confidence
                if probas[i].max() > probas[j].max():
                    index_remove.append(j)
                else:
                    index_remove.append(i)

    bboxes_scaled = bboxes_scaled.tolist()
    probas = probas.tolist()

    # Delete bounding boxes having index in index_remove
    # Sort index_remove in descending order to avoid out of range error
    index_remove.sort(reverse=True)
    for i in index_remove:
        del bboxes_scaled[i]
        del probas[i]
    
    # Convert list to tensor
    probas = torch.tensor(probas)
    return probas, bboxes_scaled

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


# Get the IoU between two bounding boxes
def Get_IoU(bbox1, bbox2):
    # bbox1 and bbox2 are two bounding boxes in the form of [xmin, ymin, xmax, ymax]
    # Get the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # Check if the two bounding boxes intersect
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate the IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)

    return iou

# Get the TP, FP, FN for each image following the IoU threshold
def Get_TP_FP_FN_byIoU(bboxes_scaled, probas, object_count, Bbox_GT, class_code_GT, IOU_threshold):
    # Initialize TP, FP, FN
    TP = 0
    FP = 0
    FN = 0

    # Get the class code of the bounding box
    class_code = Get_ClassCode(probas)

    # Convert the bounding boxes to list
    bboxes_scaled = bboxes_scaled.tolist()

    # Start compute TP, FP, FN
    for i in range(len(bboxes_scaled)):
        for j in range(len(Bbox_GT)):
            if Get_IoU(bboxes_scaled[i], Bbox_GT[j]) > IOU_threshold:
                if class_code[i] == class_code_GT[j]:
                    TP += 1
                else:
                    FP += 1
            
    # Compute FN
    FN = object_count - TP

    return TP, FP, FN
            


# Get the TP, FP, FN for each image following the distance of the center coordinates
def Get_TP_FP_FN(bboxes_scaled, probas, object_count, Bbox_GT, class_code_GT):
        # Bboxes_scaled and probas are kept before removing overlapping bounding boxes
        # BBox_GT is the ground truth bounding box along with the class code
        # object_count is the number of objects in the image from the ground truth bounding box

        # Initialize TP, FP, FN
        TP = 0
        FP = 0
        FN = 0
    
        # Get the class code of the bounding box
        class_code = Get_ClassCode(probas)
    
        # Get the center coordinates of the bounding box
        center_x = (bboxes_scaled[:, 0] + bboxes_scaled[:, 2]) / 2
        center_y = (bboxes_scaled[:, 1] + bboxes_scaled[:, 3]) / 2
    
        # Combine center_x and center_y to get the center coordinates
        center = torch.stack((center_x, center_y), dim=1)
        center = center.tolist()
    
        # Get the center coordinates of the ground truth bounding box
        # BBbox_GT is a list of bounding boxes
        # Convert Bbox_GT to tensor
        Bbox_GT = torch.tensor(Bbox_GT)
        center_x_GT = (Bbox_GT[:, 0] + Bbox_GT[:, 2]) / 2
        center_y_GT = (Bbox_GT[:, 1] + Bbox_GT[:, 3]) / 2
    
        # Combine center_x and center_y to get the center coordinates
        center_GT = torch.stack((center_x_GT, center_y_GT), dim=1)
        center_GT = center_GT.tolist()
    
        # Get the TP, FP, FN
        # Can check IoU of 2 bounding boxes as the condition to compare

        for i in range(len(center)):
            for j in range(len(center_GT)):
                dist = distance(center[i], center_GT[j])
                if dist < 40:
                    if class_code[i] == class_code_GT[j]:
                        TP += 1
                        break
                    else:
                        FP += 1
                        break
    
        FN = object_count - TP

        return TP, FP, FN

def Get_ClassCode(scores):
    # scores is a tensor
    class_code = []
    for p in scores:
        cl = p.argmax()
        cl = cl.tolist()
        class_code.append(cl)
    return class_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser('EGG Evaluation starting ...', parents=[get_args_parser()])

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Show the command line arguments
    print("Command line:")
    # Print the command line arguments
    print('python final_evaluate.py --resume' + args.resume + ' --batch_size ' + str(args.batch_size) \
        + ' --num_classes ' + str(args.num_classes)+ ' --coco_path ' + args.coco_path)
    print('----------------------------------------------------------------------------')

    print("Start time: ", datetime.datetime.now())

    transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model, criterion, postprocessors = build_model(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    DIR_TEST = os.path.join(args.coco_path, 'test')
    test_images = collect_all_images(DIR_TEST)

    count = 0

    # Initialize Global TP, FP, FN
    Global_TP = 0
    Global_FP = 0
    Global_FN = 0

    for image in test_images:
        if (count == 500):
            break
        img = Image.open(image)
        # Replace the path of the image with the path of the xml file
        xml_file = image.replace('jpg', 'xml')
        # Get the number of ground truth bounding boxes and the ground truth bounding boxes
        object_count, Bbox_GT, class_code_GT = Get_GroundTruth_Bounding_Boxes(xml_file)

        # Reset scores and boxes
        scores = []
        boxes = []
        
        print('----------------------------------------------------------------------')
        print('Image: ', image)
        scores, boxes = detect(img, model, transform)
        scores_1, boxes_1 = Select_Bounding_Boxes(scores, boxes)
        # Compute TP, FP, FN
        #TP, FP, FN = Get_TP_FP_FN(boxes, scores, object_count, Bbox_GT, class_code_GT)
        TP, FP, FN = Get_TP_FP_FN_byIoU(boxes, scores, object_count, Bbox_GT, class_code_GT, IOU_threshold=0.85)

        # Print TP, FP, FN in one line
        print("Fusion Metrics with corresponding IOU (0.5):")
        print('TP:{} FP:{} FN:{}'.format(TP, FP, FN))

        # Accumulate TP, FP, FN
        Global_TP += TP
        Global_FP += FP
        Global_FN += FN

        # Get the class of the bounding box
        classes = Get_ClassCode(scores_1)

        #print(boxes_1)
        print('Number of ground truth bounding boxes: ', object_count)
        print('Number of predicted bounding boxes: ', len(boxes_1)) 
        

        Detailed = False
        if (Detailed == True):
            print('Details of predicted bounding boxes: (Class Code + bounding boxes)')
            for i in range(len(boxes_1)):
                print(classes[i], boxes_1[i])

        count += 1

    # Print Global TP, FP, FN
    print('----------------------------------------------------------------------')
    print('Global Metrics:')
    print('Global TP:{} Global FP:{} Global FN:{}'.format(Global_TP, Global_FP, Global_FN))

    # Finish counting bounding boxes
    print('----------------------------------------------------------------------')
    print('Finish counting bounding boxes!!')
    print("Time complete counting bounding boxes: ", datetime.datetime.now())
    print('----------------------------------------------------------------------')
    print('Starting to compute mAP ...')

    # Compute mAP
    # Evaluate_AP(model, args)

    # Compute mAP50 for each class
    AP = Evaluate_AP_EachClass(model, args)

    # Print mAP
    print('-------------------------------------------------------------------------------')
    print('AP50 for each class: ', AP)

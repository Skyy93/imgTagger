from torchvision import transforms
import ast
import os
import glob
import torch

def get_std_preprocessor():
    return transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

labels = ast.literal_eval(open(os.path.join('backend', 'imagenet_labels.txt'), 'r').read())

def get_imagenet_class(logits):
    return labels[int(torch.argmax(logits))].split(',')[0]

def get_all_images(path, exts):
    files = []
    [files.extend(glob.glob(os.path.join(path, '*.' + e))) for e in exts]
    return files


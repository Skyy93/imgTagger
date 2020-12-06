from backend.utils import get_std_preprocessor, get_all_images
from backend.clf import std_clf
import torch
import os
import argparse


def main(args):

    exts = ['png', 'jpg', 'gif', 'JPEG', 'webp', 'PNG' ]
    print("Initializing preprocessor")
    preprocessor = get_std_preprocessor()
    images = get_all_images(args.in_path, exts)
    print("Initializing classifier")
    classifier = std_clf(model_type=args.model, preprocessor=preprocessor)

    classifier.process_images(images, args.out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path',
                        default='images',
                        type=str,
                        help=f'The path to the input images'
                        )
    parser.add_argument('--out_path',
                        default='images_sorted',
                        type=str,
                        help=f'Output path of the sorted images'
                        )
    parser.add_argument('--model',
                        choices=['resnet152', 'mobilenet_v2', 'resnet18', 'resnext101_32x8d'],
                        default='resnet152',
                        help=f'The pretrained backend model, resnet151 recommended due higher accuracy'
                        )
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    main(args)

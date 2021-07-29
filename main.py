import argparse
import numpy as np
import os
import sys
import time
import shutil
import random
import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms as T
import torchvision.transforms.functional as TF
import cv2
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from models.generator import Generator


def get_prediction(img_path, model, threshold=0.8):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    if len(np.where(np.array(pred_score) > threshold)[0]) > 0:
        pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
        masks = masks[:pred_t+1]
        return masks
    else:
        print('no instance in %s' % img_path); sys.exit(0)


def cartoon_transfer(args, model, image_path):
    image = Image.open(image_path).convert('RGB')
    trf = utils.get_no_aug_transform()
    image_list = [image]
    image_list = torch.from_numpy(np.array([trf(img).numpy() for img in image_list])).to(args.device)

    if args.device == 'cuda':
        model.to(args.device)

    with torch.no_grad():
        generated_images = model(image_list)
    generated_images = utils.inv_normalize(generated_images, args.device)

    pil_images = []
    for i in range(generated_images.size()[0]):
        generated_image = generated_images[i].cpu()
        pil_images.append(TF.to_pil_image(generated_image))
    return pil_images


def config():
    eval_arg = argparse.ArgumentParser()
    eval_arg.add_argument("--input_image", type=str, required=True)
    eval_arg.add_argument("--output_dir", type=str, default="output")
    eval_arg.add_argument("--pretrained", type=str, default="checkpoints/trained_netG.pth")
    args = eval_arg.parse_args()
    return args


if __name__ == '__main__':
    args = config()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    segmentator = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    segmentator.eval()

    transfer = Generator()
    transfer.load_state_dict(torch.load(args.pretrained))
    transfer.eval()
    
    img_path = args.input_image
    masks = get_prediction(img_path, segmentator)
    
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    img = cv2.imread(img_path)
    cv2.imwrite(args.output_dir+'/%s' % (args.input_image), img)

    img_style = cartoon_transfer(args, transfer, img_path)[0]
    img_style = np.asarray(img_style)
    img_style = cv2.resize(img_style, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    img_style = cv2.cvtColor(img_style, cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.output_dir+'/%s_transfered.jpg' % (args.input_image.replace('.jpg','')), img_style)
    img = img_style

    for i in range(len(masks)):
      mask = np.where(masks[i] == 1, 255, 0).astype(np.uint8)
      img_mask = cv2.bitwise_and(img, img, mask=mask)
      wbg = np.ones_like(img, np.uint8)*255
      wbg = cv2.bitwise_not(wbg, wbg, mask=mask)
      img_mask = wbg + img_mask
      
      y, x = masks[i].nonzero()
      minx = np.min(x)
      miny = np.min(y)
      maxx = np.max(x)
      maxy = np.max(y)
      img_crop = img_mask[miny:maxy, minx:maxx, :]

      img_crop = cv2.cvtColor(img_crop, cv2.COLOR_RGB2RGBA)
      img_crop[:, :, 3] = 255
      whiteCellsMask = np.logical_and(img_crop[:,:,0] == 255, np.logical_and(img_crop[:,:,1] == 255, img_crop[:,:,2] == 255))
      img_crop[whiteCellsMask, :] = [255, 255, 255, 0]
      cv2.imwrite(args.output_dir+'/%s_%s.png' % (args.input_image.replace('.jpg', ''),i), img_crop)


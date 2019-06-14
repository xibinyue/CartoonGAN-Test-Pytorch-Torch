#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - xibin.yue <xibin.yue@moji.com>
import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer
import zerorpc
from log_utils import MyLogger

valid_ext = ['.jpg', '.png']


def init_model():
    model = Transformer()
    model.load_state_dict(torch.load(os.path.join('./pretrained_model', 'Hayao' + '_net_G_float.pth')))
    model.eval()
    model.cuda(0)
    return model


cartoon_model = init_model()


class CartoonChange(object):
    def __init__(self, model_def=cartoon_model):
        self.model = model_def

    def change_to_cartoon(self, img_url, dst_dir):
        input_image = Image.open(img_url).convert("RGB")
        h = input_image.size[0]
        w = input_image.size[1]
        ratio = h * 1.0 / w
        if ratio > 1:
            h = 450
            w = int(h * 1.0 / ratio)
        else:
            w = 450
            h = int(w * ratio)
        input_image = input_image.resize((h, w), Image.BICUBIC)
        input_image = np.asarray(input_image)
        # RGB -> BGR
        input_image = input_image[:, :, [2, 1, 0]]
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        input_image = -1 + 2 * input_image
        input_image = Variable(input_image, volatile=True).cuda()

        # forward
        output_image = self.model(input_image)
        output_image = output_image[0]
        # BGR -> RGB
        output_image = output_image[[2, 1, 0], :, :]
        # deprocess, (0, 1)
        output_image = output_image.data.cpu().float() * 0.5 + 0.5
        # save
        dst_url = os.path.join(dst_dir, img_url.split('/')[-1])
        vutils.save_image(output_image, dst_url)
        return dst_url


def main():
    s = zerorpc.Server(CartoonChange())
    s.bind("tcp://0.0.0.0:4242")
    s.run()


if __name__ == '__main__':
    main()

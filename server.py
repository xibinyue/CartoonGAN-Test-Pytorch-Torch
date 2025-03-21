#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 - xibin.yue <xibin.yue@moji.com>
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer
import zerorpc
import requests
from tomorrow import threads
import time
from sqlalchemy import create_engine
import pandas as pd
import cv2

sns_engine = create_engine('mysql+mysqldb://snsro1:CpuHli03@192.168.10.18:3337/snsforum?charset=utf8')

valid_ext = ['.jpg', '.png']
base_dir = 'http://oss4liview.moji.com'
local_dir = 'sns_imgs'
cartoon_out_dir = 'cartoon_out'

size1 = (320, 505)
size2 = (160, 252)


def init_model():
    model = Transformer()
    model.load_state_dict(torch.load(os.path.join('./pretrained_model', 'Hayao' + '_net_G_float.pth')))
    model.eval()
    model.cuda(0)
    return model


cartoon_model = init_model()


def resize_image(src_path, size, size_index):
    img = cv2.imread(src_path)
    img_resize = cv2.resize(img, size)
    dst_path = src_path.replace('.jpg', '_%s.png' % size_index)
    cv2.imwrite(dst_path, img_resize)
    return dst_path


# @threads(20)
def download_one_img(url, img_path):
    try:
        binary_data = requests.get(url)
        temp_file = open(img_path, 'wb')
        temp_file.write(binary_data.content)
        temp_file.close()
        return True
    except Exception as e:
        print 'image download error: %s[%s][%s]' % (e, url, img_path)
    return False


class CartoonChange(object):
    def __init__(self, model_def=cartoon_model):
        self.model = model_def

    def get_img_urls(self):
        urls = []
        stamp_start = int(time.time()) - 10 * 60 * 1000
        sql = "SELECT * FROM picture_base WHERE sns_id=56587715  AND upload_time > '%d'ORDER BY upload_time DESC limit 3" % stamp_start
        res = pd.read_sql(sql, sns_engine)
        res['oss_path'] = res['path'].map(lambda x: os.path.join(base_dir, x))
        for i in range(len(res)):
            oss_path = res['oss_path'].iloc[i]
            dst_path = os.path.join(local_dir, oss_path.split('/')[-1])
            print oss_path
            download_one_img(oss_path, dst_path)
            cartoon_dst_path = self.change_to_cartoon(dst_path, cartoon_out_dir)
            urls.append(os.path.join(os.getcwd(), cartoon_dst_path))

        small = []
        big = []
        for url in urls:
            big_path = resize_image(url, size1, 'big')
            small_path = resize_image(url, size2, 'small')
            small.append(small_path)
            big.append(big_path)
        return {'small': small, 'big': big}

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
    # # get_img_urls()
    # cartoon = CartoonChange()
    # print cartoon.get_img_urls()

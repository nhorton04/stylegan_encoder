import numpy as np
import os
import pickle
import streamlit as st
import sys
import tensorflow as tf
import urllib

from align import align
from encode import main

import config
import dnnlib
import dnnlib.tflib as tflib
import pickle
import PIL.Image
import bz2
import cv2
import argparse
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector2 import LandmarksDetector
import multiprocessing

import pandas as pd
from PIL import Image


# This should not be hashed by Streamlit when using st.cache.
TL_GAN_HASH_FUNCS = {
    tf.Session : id
}

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path



def main():
    #Upload images
    uploaded_file = st.file_uploader("Choose a picture", type=['jpg', 'png'])
    if uploaded_file is not None:
        st.image(uploaded_file, width=200)
    second_uploaded_file = st.file_uploader("Choose another picture", type=['jpg', 'png'])
    if second_uploaded_file is not None:
        st.image(second_uploaded_file, width=200)

    img1 = np.array(uploaded_file)
    img2 = np.array(second_uploaded_file)

    image1 = PIL.Image.open(uploaded_file)
    image2 = PIL.Image.open(second_uploaded_file)

    images = [image1, image2]

    if st.button('Align images'):
        align(images)



def align(images):
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    RAW_IMAGES_DIR = './raw_images'
    ALIGNED_IMAGES_DIR = './aligned_images'

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in os.listdir(RAW_IMAGES_DIR):
        print('Aligning %s ...' % img_name)
        try:
            raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            fn = face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], 1)
            if os.path.isfile(fn):
                continue
            print('Getting landmarks...')
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                try:
                    print('Starting face alignment...')
                    face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                    image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=1024, x_scale=1, y_scale=1, em_scale=0.1, alpha=False)
                    print('Wrote result %s' % aligned_face_path)
                except:
                    print("Exception in face alignment!")
        except:
            print("Exception in landmark detection!")



if __name__ == "__main__":
    main()

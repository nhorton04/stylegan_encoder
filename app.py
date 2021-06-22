import numpy as np
import os
import pickle
import streamlit as st
import sys
import tensorflow as tf
import urllib

from align import align
from encode import encode

import config
import dnnlib
import dnnlib.tflib as tflib
import pickle
import PIL.Image
# import cv2
# import argparse
import numpy as np
import config
import dnnlib
import dnnlib.tflib as tflib
import pickle
import PIL.Image
# sys.path.append('tl_gan')
# sys.path.append('pg_gan')
# import feature_axis
# import tfutil
# import tfutil_cpu

# This should not be hashed by Streamlit when using st.cache.
TL_GAN_HASH_FUNCS = {
    tf.compat.v1.InteractiveSession: id
}


def main():

    uploaded_file = st.file_uploader(
        "Choose first picture", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        st.image(uploaded_file, width=200)
    else:
        uploaded_file = os.path.abspath(os.getcwd()) + '/raw_images/Gunny.jpg'
        st.image(uploaded_file, width=50)

    second_upload = st.file_uploader(
        "Choose second picture", type=['jpg', 'jpeg', 'png'])
    if second_upload is not None:
        st.image(second_upload, width=200)
    else:
        second_upload = os.path.abspath(os.getcwd()) + '/raw_images/me.png'
        st.image(second_upload, width=50)

    images = [uploaded_file, second_upload]

    ALIGNED_IMAGES_DIR = os.getcwd() + '/aligned_images'
    RAW_IMAGES_DIR = os.getcwd() + '/raw_images'

    align()
    encode()

    # load the StyleGAN model into Colab
    URL_FFHQ = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
    tflib.init_tf()
    with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)
    # load the latents
    s1 = np.load('latent_representations/IMG_8124_01.npy')
    s2 = np.load('latent_representations/IMG_20170914_154355_01.npy')
    s3 = np.load('latent_representations/225809_8815259850_4603_n_01.npy')
    s1 = np.expand_dims(s1, axis=0)
    s2 = np.expand_dims(s2, axis=0)
    s3 = np.expand_dims(s3, axis=0)
    # combine the latents somehow... let's try an average:

    x = st.slider('picture 1', 0.01, 0.99, 0.33)
    y = st.slider('picture 2', 0.01, 0.99, 0.33)
    z = st.slider('picture 3', 0.01, 0.99, 0.33)
    savg = (x*s1+y*s2+z*s3)

    # run the generator network to render the latents:
    synthesis_kwargs = dict(output_transform=dict(
        func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
    images = Gs_network.components.synthesis.run(
        savg, randomize_noise=False, **synthesis_kwargs)

    for image in images:
        st.image((PIL.Image.fromarray(images.transpose((0, 2, 3, 1))
                 [0], 'RGB').resize((512, 512), PIL.Image.LANCZOS)))

    for image in images:
        st.image(image, width=200)


# @st.cache(allow_output_mutation=True, hash_funcs=TL_GAN_HASH_FUNCS)
# def load_model():
#     """
#     Create the tensorflow session.
#     """
#     # Open a new TensorFlow session.
#     config = tf.ConfigProto(allow_soft_placement=True)
#     session = tf.Session(config=config)
#
#     # Must have a default TensorFlow session established in order to initialize the GAN.
#     with session.as_default():
#         # Read in either the GPU or the CPU version of the GAN
#         with open(MODEL_FILE_GPU if USE_GPU else MODEL_FILE_CPU, 'rb') as f:
#             G = pickle.load(f)
#     return session, G
if __name__ == "__main__":
    main()

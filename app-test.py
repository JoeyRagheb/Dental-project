import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow import convert_to_tensor
import numpy as np
import torch
from torchvision.transforms import Resize
import torch as T
from functools import reduce
import pm4py
from torchvision import transforms
import torchvision
import tensorflow
import cv2
import os
from ultralytics import YOLO
import glob


def processed_img(img_path):
    img = load_img(img_path, target_size=(960, 960, 3))

    st.info('***Image segmentation in the works***')
    os.system('yolo task=segment mode=predict model=weights/best.pt conf=0.25 source=test-image/ save=true')

    # st.image('runs/segment/predict/*.jpg')
    st.info('***Hooray! Segmentation Result***')
    for image_path in glob.glob(f'runs/segment/predict/*.jpg')[:3]:
        st.image(image_path)

    # st.info(img)

    # img = img_to_array(img)
    # tensor_image = torch.from_numpy(img)
    # tensor_image = tensor_image.permute(2, 0, 1)
    # tensor_image = tensor_image.unsqueeze(0)
    # img = tensor_image
    # st.info("***THISSS***")
    # st.info(type(img))
    # f = img.shape
    # st.info(f)



    # torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    # #Model
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='Best/best.pt', force_reload=True)

    #Inference
    # results = model(img)
    # results = model(img)

    return results

def run():
    st.title("Maha Cavity Finder")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            result = processed_img(save_image_path)

            # st.image(result)

            

run()
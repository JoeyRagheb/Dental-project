import streamlit as st
from PIL import Image
# from tensorflow.keras.utils import load_img
import numpy as np
import torch
import os
from ultralytics import YOLO
import glob
import re
import subprocess
import streamlit_authenticator as stauth

st.cache_data.clear()



def processed_img(img_path):
    # img = load_img(img_path, target_size=(960, 960, 3))
    img = Image.open(img_path).resize((960, 960))

    st.info('***Image segmentation in the works***')
    # source_folder = os.path.dirname(img_path) + '/'
    source_folder = img_path
    
    os.makedirs('runs/segment/', exist_ok=True)
    
    
    os.system(f'yolo task=segment mode=predict model=weights/best.pt conf=0.25 source={source_folder} save=true')
    

    folders = os.listdir('runs/segment/')
    folders = [re.findall(r'\d+', s) for s in folders]
    folders = [int(num) for sublist in folders for num in sublist]
    folders.sort()
    # st.info(folders)
    if folders != []:
        folder_number = folders[-1]
    else:
        folder_number=''
    image_pred = 'runs/segment/predict' + str(folder_number) + '/*.jpg'
    # st.info(image_pred)
    
    image_predictions =[]
    st.info('***Success! Segmentation Result***')
    
    for image_path in glob.glob(image_pred):
        image_predictions += [image_path]

    st.image(image_predictions[-1], width=400)
    

    folder_path = 'runs/segment/'
    os.system(f"rm -r {folder_path}")
    # st.info(f"Folder {folder_path} deleted successfully.")


    results = 0
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
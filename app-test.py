import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img
# from tensorflow.keras.utils import img_to_array
# from tensorflow import convert_to_tensor
import numpy as np
import torch
# from torchvision.transforms import Resize
# import torch as T
# from functools import reduce
# import pm4py
# from torchvision import transforms
# import torchvision
# import tensorflow
# import cv2
import os
from ultralytics import YOLO
import glob
import re
import subprocess
st.cache_data.clear()



def processed_img(img_path):
    img = load_img(img_path, target_size=(960, 960, 3))

    st.info('***Image segmentation in the works***')
    # source_folder = os.path.dirname(img_path) + '/'
    source_folder = img_path
    os.system(f'yolo task=segment mode=predict model=weights/best.pt conf=0.25 source={source_folder} save=true')
    
    # # Define the YOLO command
    # yolo_command = "yolo task=segment mode=predict model=weights/best.pt conf=0.25 source=test-image/ save=true"

    # # Execute the YOLO command and capture the output
    # output = subprocess.check_output(yolo_command, shell=True)
    # st.info('***HERE---------------------------***')
    # st.info(output)

    # # Process the output as needed
    # # Here you can save the output to a variable or perform any further actions

    # # # For example, if the output is an image file path
    # image_path = output.decode().strip()  # Convert the output bytes to a string and remove leading/trailing whitespace
    # st.info(image_path)

    # # # Now you can use the image path or further process the image as needed
    # # print(image_path)
    





    
    folders = os.listdir('runs/segment/')
    folders = [re.findall(r'\d+', s) for s in folders]
    folders = [int(num) for sublist in folders for num in sublist]
    folders.sort()
    st.info(folders)
    if folders != '':
        folder_number = folders[-1]
    else:
        folder_number=''
    image_pred = 'runs/segment/predict' + str(folder_number) + '/*.jpg'
    st.info(image_pred)
    
    image_predictions =[]
    st.info('***Hooray! Segmentation Result***')
    
    for image_path in glob.glob(image_pred):
        image_predictions += [image_path]

    st.image(image_predictions[-1])

    # pic_t = os.listdir('runs/segment/predict8/')
    # st.info(pic_t)
    # st.info(image_predictions)
    # st.image(image_predictions[-1])
    


    # Specify the path to the folder
    folder_path = 'runs/segment/'

    # List to store file paths
    file_paths = []

    # Iterate through all files and subdirectories
    for root, directories, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))

    all_files = []
    # Print the result
    for file_path in file_paths:
        all_files+=[file_path]
    st.info(all_files)



    

    folder_path = 'runs/segment/'
    os.system(f"rm -r {folder_path}")
    st.info(f"Folder {folder_path} deleted successfully.")


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


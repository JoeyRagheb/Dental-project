import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow import convert_to_tensor
import numpy as np
import torch


# from keras.models import load_model
# import requests
# from bs4 import BeautifulSoup

# vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
#               'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
#               'Tomato', 'Turnip']


# def fetch_calories(prediction):
#     try:
#         url = 'https://www.google.com/search?&q=calories in ' + prediction
#         req = requests.get(url).text
#         scrap = BeautifulSoup(req, 'html.parser')
#         calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
#         return calories
#     except Exception as e:
#         st.error("Can't able to fetch the Calories")
#         print(e)


def processed_img(img_path):
    # img = load_img(img_path)
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)

    tensor_image = torch.from_numpy(img)
    tensor_image = tensor_image.permute(2, 0, 1)
    tensor_image = tensor_image.unsqueeze(0)


    # img = img_to_array(img)
    # img = np.moveaxis(img, -1, 0)
    # img = np.expand_dims(img, [0])
    # img = img / 255
    # img = np.expand_dims(img, [0])

    img = tensor_image
    
    

    # import torch

    #Model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='Best/best.pt', force_reload=True)
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/yolov5s.pt', force_reload=True)
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt', force_reload=True)
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

    # #Images
    # img = DataLoader()

    #Inference
    results = model(img)

    # #Results
    # results.print() # or .show(), .save(), .crop(), .pandas(), etc.
    # results.xywh[0]




    # res ="carrot"
    # # res should be a string
    # print(res)
    # return res.capitalize()
    return results

def run():
    st.title("Maha Cavity Finder")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        # save_image_path = './' + img_file.name
        save_image_path = img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            result = processed_img(save_image_path)
            print(result)
            # if result in vegetables:
            #     st.info('**Category : Vegetables**')
            # else:
            #     st.info('**Category : Fruit**')
            # st.success("**Predicted : " + result + '**')
            # # cal = fetch_calories(result)
            # # if cal:
            # #     st.warning('**' + cal + '(100 grams)**')
            
            st.info('**TESTING**')
            # result = result.permute(1, 2, 0).numpy()
            # st.info(result)
            # # result = result.numpy()
            # # result = np.squeeze(result)
            # st.image(result)


run()
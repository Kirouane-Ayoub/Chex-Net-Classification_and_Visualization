from model.chest_model import *
import pandas as pd
import streamlit as st
st.spinner(text="In progress...")
model = Modelchest()
df = pd.read_csv('train-small.csv')
IMAGE_DIR = os.getcwd()
labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']
with st.sidebar : 
    st.image("icon.png" , width=250)
    Prediction = st.radio("Select Type of Prediction : " , ("Classification" , "Classification  and Visualization"))

tab1 , tab2 = st.tabs(["Home" , "Detection"])

with tab1 : 
    st.header('About the Project and the Dataset : ')
    st.image("ezgif-3-7977ed06aa.gif")
    st.write("""Chest X-rays are currently the best available method for diagnosing pneumonia, playing a crucial role in clinical care and epidemiological studies.
              Pneumonia is responsible for more than 1 million hospitalizations and 50,000 deaths per year in the US alone.""")
    st.write(""" 
    CheXNet is a convolutional neural network (CNN) designed for automated interpretation of chest radiographs, developed by Stanford researchers.
    It was trained to classify 14 different pathologies on chest X-rays, including atelectasis, cardiomegaly, consolidation, edema, effusion, emphysema, fibrosis, hernia, infiltration, mass, nodule, pleural thickening, pneumothorax, and normal.

    The CheXNet model architecture is based on a deep residual network (ResNet) with 121 layers. 
    The model was trained using a dataset of over 100,000 chest X-ray images with labels from the National Institutes of Health (NIH) Clinical Center. 
    The dataset was manually labeled by radiologists and used as a benchmark for the CheXNet model's performance.

    CheXNet achieved state-of-the-art performance on the NIH dataset, outperforming radiologists in identifying certain pathologies. 
    The model's ability to automate the interpretation of chest X-rays has the potential to improve healthcare outcomes by reducing the time and cost of manual radiological interpretation and increasing diagnostic accuracy.
    """)


with st.spinner('This will take some time...'):
    with tab2 : 
        file = st.file_uploader("Select Your Xray Image : " , type=["jpeg" , "png" , "jpg"])
        if file : 
            img_name = file.name
            img_save = img_name.split(".")[0]
            st.image(img_name , caption="The Originale Image")
        if st.button("Click to start") :
            if Prediction == "Classification" : 
                preds = predection(img_name , IMAGE_DIR , model , df)
                dataf = pred_dataframe(preds , labels)
                st.dataframe(dataf)
            else : 
                compute_gradcam(model ,img_name, IMAGE_DIR , df , labels , labels[:2])
                st.header("Result : ")
                col1 , col2  = st.columns(2)
                folder_path = os.path.join(IMAGE_DIR ,img_save)
                imglist = os.listdir(folder_path)
                slice = int(len(imglist) / 2)
                with col1:
                    for img in imglist[:slice] : 
                        impath = f"{folder_path}/{img}"
                        st.image(impath)
                with col2 :
                    for img in imglist[slice : ] : 
                        impath = f"{folder_path}/{img}"
                        st.image(impath)
        st.success('Done!')
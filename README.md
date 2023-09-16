# Chex-Net
<img src=https://github.com/Kirouane-Ayoub/chex-Net-Classification_and_Visualization/assets/99510125/f000c461-ed51-4a36-a2a9-0eb638d03417/>

## Overview
**CheXNet** is a convolutional neural network (CNN) designed for automated interpretation of chest radiographs, developed by Stanford researchers. It was trained to classify 14 different pathologies on chest X-rays, including atelectasis, cardiomegaly, consolidation, edema, effusion, emphysema, fibrosis, hernia, infiltration, mass, nodule, pleural thickening, pneumothorax, and normal.

The **CheXNet** model architecture is based on a deep residual network (ResNet) with 121 layers. The model was trained using a dataset of over 100,000 chest X-ray images with labels from the National Institutes of Health (NIH) Clinical Center. The dataset was manually labeled by radiologists and used as a benchmark for the CheXNet model's performance.

CheXNet achieved state-of-the-art performance on the NIH dataset, outperforming radiologists in identifying certain pathologies. The model's ability to automate the interpretation of chest X-rays has the potential to improve healthcare outcomes by reducing the time and cost of manual radiological interpretation and increasing diagnostic accuracy.
**https://stanfordmlgroup.github.io/projects/chexnet**
## How To Run This APP

```
pip install -r requirements.txt
streamlit run app.py
```

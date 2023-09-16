# Chex-Net
<img src=https://github.com/Kirouane-Ayoub/chex-Net-Classification_and_Visualization/assets/99510125/f000c461-ed51-4a36-a2a9-0eb638d03417/>

## Overview
**CheXNet** is a convolutional neural network (CNN) designed for automated interpretation of chest radiographs, developed by Stanford researchers. It was trained to classify 14 different pathologies on chest X-rays, including atelectasis, cardiomegaly, consolidation, edema, effusion, emphysema, fibrosis, hernia, infiltration, mass, nodule, pleural thickening, pneumothorax, and normal.

The **CheXNet** model architecture is based on a deep residual network (ResNet) with 121 layers. The model was trained using a dataset of over 100,000 chest X-ray images with labels from the National Institutes of Health (NIH) Clinical Center. The dataset was manually labeled by radiologists and used as a benchmark for the CheXNet model's performance.

CheXNet achieved state-of-the-art performance on the NIH dataset, outperforming radiologists in identifying certain pathologies. The model's ability to automate the interpretation of chest X-rays has the potential to improve healthcare outcomes by reducing the time and cost of manual radiological interpretation and increasing diagnostic accuracy.
**https://stanfordmlgroup.github.io/projects/chexnet**
## Frameworks
* **Pandas** is a library for data manipulation and analysis. It provides a number of functions for working with data **frames**, which are two-dimensional data structures that can be used to store and manipulate data.
* **Streamlit** is a library for building web apps. It can be used to create data visualization apps that allow users to interact with data in real-time.
* **TensorFlow** is a library for machine learning. It provides a number of functions for building and training machine learning models.
* **NumPy** is a library for scientific computing. It provides a high-performance array data type and a number of functions for mathematical operations.
* **OpenCV-Python** is a library for computer vision. It provides a wide range of functions for image processing, object detection, and machine learning.
* **Matplotlib** is a library for plotting data. It provides a number of functions for creating graphs and charts.
## How To Run This APP

```
pip install -r requirements.txt
streamlit run app.py
```

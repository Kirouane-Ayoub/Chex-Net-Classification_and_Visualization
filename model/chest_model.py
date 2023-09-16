import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np 
import pandas as pd

def get_mean_std_per_batch(image_path, df, H=320, W=320):
    sample_data = []
    for idx, img in enumerate(df.sample(100)["Image"].values):
        # path = image_dir + img
        sample_data.append(
            np.array(image.load_img(image_path, target_size=(H, W))))

    mean = np.mean(sample_data[0])
    std = np.std(sample_data[0])
    return mean, std


def Modelchest() : 
    base_model = DenseNet121(include_top = False , pooling='avg')
    x = base_model.output
    predections = Dense(14 , activation="sigmoid")(x)
    model = Model(inputs=base_model.input , outputs=predections)
    model.load_weights('xr14m/pretrained_model.h5')
    return model 

def load_image(img, image_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    img_path = f"{image_dir}/{img}"
    mean, std = get_mean_std_per_batch(img_path, df, H=H, W=W)
    x = image.load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x




def compute_gradcam(model, img, image_dir, df, labels, selected_labels, layer_name='bn',W = 224, H=224):
    
    omg_name = img.split(".")[0]
    os.mkdir(str(omg_name))
    preprocessed_input = load_image(img, image_dir, df)
    predictions = model.predict(preprocessed_input)    
    ##############################
    print("Loading original image")
    plt.figure(figsize=(50, 50))
    plt.subplot(1,1,1 , label="original")
    #plt.title("Original")
    #plt.axis('off')
    #plt.imshow(load_image(img, image_dir, df, preprocess=False), cmap='gray')
    ##############################
    
    
    layer_name='bn'
    conv_output = model.get_layer(layer_name).output
    gradModel = Model(
                inputs=[model.inputs],
                outputs=[conv_output,model.output])
    
    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")

            cls = 0 # specific class output probability
            with tf.GradientTape() as tape:
                (convOutputs, pred) = gradModel(preprocessed_input)
                loss = pred[:, cls]
            # use automatic differentiation to compute the gradients
            grads = tape.gradient(loss, convOutputs)
            
            output, grads_val = convOutputs[0, :], grads[0, :, :, :] #no need of batch information

            weights = np.mean(grads_val, axis=(0, 1))
            cam = np.dot(output, weights)

            # Process CAM
            cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
            cam = np.maximum(cam, 0)
            gradcam = cam / cam.max()
            
            ###############################
            plt.subplot(1,1,1 , label="class")
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}" , fontsize=180)
            plt.axis('off')
            plt.imshow(load_image(img, image_dir, df, preprocess=False),cmap='gray')

            value = np.array(min(0.5, predictions[0][i])).reshape(1,1)
            value = min(0.5, predictions[0][i])
            value = np.repeat(value,W*H).reshape(W,H)
            plt.imshow(gradcam, cmap='jet', alpha=value)
            plt.savefig(f"{omg_name}/{labels[i]}.png" , format="png")
            
            j += 1
            #################################


def predection(img_name , IMAGE_DIR , model , df):
    x = load_image(img=img_name, image_dir=IMAGE_DIR , df=df)
    preds = model.predict(x)
    return preds

def pred_dataframe(preds , labels) : 
    pred_df = pd.DataFrame(preds , columns=labels)
    return pred_df
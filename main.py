import os
import numpy as np
# import cv2
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array


app = Flask(__name__)
existing_model = tf.keras.models.load_model('ensemble_model.hdf5', compile = False)

def predict_skin_cancer(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Adjust target_size as needed
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    predictions = existing_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = predicted_class_index
    return predicted_class_name

print(predict_skin_cancer("image1.jpeg"))
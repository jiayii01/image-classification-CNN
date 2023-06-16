import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Tensorflow Intel Image Classification")

@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model('intel_image_model.hdf5')
    return model

HEIGHT = 224
WIDTH = 224

def predict_class(image, model):
   
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [HEIGHT, WIDTH]) / 255
    image = np.expand_dims(image, axis = 0)

    prediction = model.predict(image)
    return prediction

model = load_model()

img_file = st.file_uploader(
    "Upload an image of buildings, forests, glaciers, mountains, sea or streets.", type=["jpg", "jpeg", "png"])

if img_file:
    slot = st.empty()
    with st.spinner("Processing Image..."):
        image = Image.open(img_file)
        st.image(image, caption="Input Image", width = 400)

        pred = predict_class(np.asarray(image), model)
                
        class_names = ['buildings', 'forests', 'glaciers', 'mountains', 'sea', 'streets']
        
        result = class_names[np.argmax(pred)]
                
        output = 'The image is of ' + result

        st.success(output)
else:
    st.text('Waiting for an image upload')

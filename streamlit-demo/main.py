import streamlit as st
import numpy as np 
import tensorflow as tf

def model_prediction(test_image):
    model=tf.keras.models.load_model("cat_or_dog_model.h5")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    predictions=model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title('hello')
app_mode=st.sidebar.selectbox("select page",["Home","About","prediction"])

if(app_mode=="prediction"):
    st.header("cat or dog")
    test_image=st.file_uploader("Choose as image")

    if(st.button("show image")):
        st.image(test_image)
    if(st.button("predict")):
        st.write("prediction")
        result_index=model_prediction(test_image)

        if(result_index==0):
            pred="cat"
        else:
            pred="dog"

        st.success("It is a "+pred)


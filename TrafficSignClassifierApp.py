
# Import Libraries
import streamlit as st
import numpy as np
from PIL import Image
import os
import time
import pickle
import matplotlib.pyplot as plt
import cv2
# import tensorflow
from tensorflow import keras
# from keras.models import load_model

# Define class labels
CLASS_LABELS = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

# Declare variables
predicted =0;
test_image =0;

# Declare all functions

# Function on read and convert image to numpy array
def readImage(image):
    img = Image.open(image)
    return img


# Load model pickles
def loadModel(path):
    keras.backend.clear_session()
    model = keras.models.load_model(path)
    return model

# Apply transformations
def transformImage(isFlipped, rotation_range, test_image):
    test_image = test_image.rotate(rotation_range)
    if isFlipped:
        test_image = test_image.transpose(Image.FLIP_LEFT_RIGHT)
#     expander.image(test_image)
    return test_image


# Function to classify image and return prediction
def classifyImage(model_name, path, image):
    
    model = loadModel(path)
    st.write(model_name, " successfully loaded!")
     # Resize image to match input size of model
    image_resized = image.resize((32,32))
    # Convert image into a numpy array
    img_arr = np.array(image_resized)
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img_arr,cv2.COLOR_RGB2GRAY)
    # Equalize and normalize image
    img_equ  = cv2.equalizeHist(img_gray)
    img_norm = img_equ/255
    


    # reshape array to match the input shape of the model
    img_norm = np.expand_dims(img_norm, axis=0)
    img_norm = np.reshape(img_norm, (
    img_norm.shape[0], img_norm.shape[1], img_norm.shape[2],1))
    # Predict image using model
    predicted = model.predict(img_norm)[0]
#     predicted_class = model.predict_classes(img_norm)[0]
    # Display results
    # Predicting image
#     st.subheader(f"Predicted class: {CLASS_LABELS[predicted_class]}")
    with st.spinner('Processing image...'):
        time.sleep(2)
    st.image(img_equ)
    displayExtra(predicted)
    return predicted
    
    
def displayExtra(predicted):
    
    progress_text = "Calculating Top 5 predictions..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.02)
    my_bar.progress(percent_complete + 1, text=progress_text)
    # List of possible predictions
    # Get the top 5 predicted classes
    top_5_indices = np.argsort(predicted)[-5:][::-1] 


    # Get the corresponding class labels
    top_5_labels = [CLASS_LABELS[i] for i in top_5_indices]


    print(top_5_indices)

    top_5_probs = predicted[top_5_indices]* 100 # get the probabilities of top 5 predictions in percentage

    rows = []
    for i in range(len(top_5_labels)):
        rows.append({
        'Label': top_5_labels[i],
        'Probability': f"{top_5_probs[i]:.2f}%"
    })

    # Display the table
    st.subheader("Top 5 predictions:")
    st.table(rows)

    
## Functions end

# Title
st.title("Traffic Sign Classifier")

# Additional HTML
html_test = """


<div style="background-color:green;padding:10px">
<h1 style="color:blue">Html test</h1>
</div>

"""

# Run HTML code
# st.markdown(html_test,unsafe_allow_html=True)
# st.markdown(styles,unsafe_allow_html=True)

# Multipage things
st.sidebar.success("Select a page.")

# Load Test images for selection box
IMG_DATA_TEST = []
IMG_NAMES = []
# Get absolute path of directory containing the file
file_dir = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.join(file_dir, 'test_images')
for img in os.listdir(test_path):
    im = Image.open(test_path +'/'+ img)
    IMG_NAMES.append(img)
    im = np.array(im)
    IMG_DATA_TEST.append(im)
IMG_DATA_ARR_TEST = np.array(IMG_DATA_TEST)


# Upload image as input
input_file = st.file_uploader(label="Upload an image", type=['.jpg', '.png'])

# Sidebar Functionality
with st.sidebar:
    option = st.selectbox(
        'Select an image',
    options =IMG_NAMES )
    select_image = Image.open(test_path +'/'+ option)

if input_file is not None:
    image = readImage(input_file)
    st.write("## Image you have selected")
    image = image.resize((128,128))
    st.image(image)
    test_image = image
else:
    st.write("## Image you have selected")
    select_image = select_image.resize((128,128))
    st.image(select_image)
    test_image = select_image
    
# Image tranformations
# Display Output
expander = st.expander("Add Image Transformations")

# Get user input for rotation degree
number = expander.number_input('Enter rotation degree')

# Get user input for horizontal flip
isFlipped = expander.checkbox('Flip Horizontal')

# Apply user input
# if expander.button('Transform Image'):
#     test_image = transformImage(isFlipped, number, test_image)
#     st.image(test_image)
    

# Allow user to pick a model
model = st.radio(
    "Pick a model to classify your image",
    ('CNN Version 1: Using RGB Images', 'CNN Version 2: Using Greyscale Images','CNN version 3: Tuned', 'CNN version 4: Augmented Data'))

if model == 'CNN Version 1: Using RGB Images':
    # Store model name in a variable
    model_name = model
    st.write('You selected ', model,' with a training accuracy of 80%')
elif model == 'CNN Version 2: Using Greyscale Images':
    model_name = model
    st.write('You selected ', model,' with a Training accuracy of 90%')
elif model == 'CNN version 4: Augmented Data':
    model_name = model
    st.write('You selected ', model,' with a Training accuracy of 94%')

    
# Take path of model as user input
path_input = st.text_input(
        "Enter model path from your directory")
path_input = path_input.replace('"', '')


# Button to classify image image
if st.button('Classify image'):
    st.write('Classifying image....')
    test_image = transformImage(isFlipped, number, test_image)
    st.image(test_image)
    predicted = classifyImage(model_name, path_input,test_image)
    
#     if 'image' in globals():
#         predicted = classifyImage(model_name, path_input,image)
#     else:
#         predicted = classifyImage(model_name, path_input,test_image)
        

# Flip Horizontal
# Rotate 
# Brightness


# Sharpen/Blur
# Stretch
# Translation




import cv2
import numpy as np
import os
import streamlit as st

def load_dataset():
    data_dir = 'data_beras'
    categories = ['basmati', 'ir64', 'pandanwangi', 'rojolele']
    data = []
    labels = []

    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                data.append(img)
                labels.append(class_num)

    if data and labels:
        data = np.array(data) / 255.0  # Normalize
        labels = np.array(labels)
        return data, labels
    else:
        return None, None

def classify_uploaded_image(model, flatten_image=False):
    import streamlit as st
    
    # Define category names
    categories = ['basmati', 'ir64', 'pandanwangi', 'rojolele']
    
    # File upload input
    uploaded_file = st.file_uploader("Upload a rice grain image for prediction...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and preprocess image
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        img_resized = cv2.resize(img, (128, 128)) / 255.0
        
        # For non-CNN models, we flatten the image
        if flatten_image:
            img_resized_flattened = img_resized.flatten().reshape(1, -1)  # Flatten the image
            prediction = model.predict(img_resized_flattened)
        else:
            # For CNN models, keep the 4D shape
            img_resized = img_resized.reshape(1, 128, 128, 3)
            prediction = np.argmax(model.predict(img_resized))
        
        # Display the uploaded image
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Convert prediction index to category name
        predicted_category = categories[prediction[0]] if flatten_image else categories[prediction]
        
        # Display the prediction
        st.write(f"Predicted Category: {predicted_category}")

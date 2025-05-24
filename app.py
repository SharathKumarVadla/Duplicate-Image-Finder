import streamlit as st
import os
import shutil
import numpy as np
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load pretrained ResNet50 model without the top layer
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model.trainable = False

def extract_feature_vector_tf(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features

def cosine_sim_tf(img1, img2):
    vec1 = extract_feature_vector_tf(img1)
    vec2 = extract_feature_vector_tf(img2)
    return cosine_similarity(vec1, vec2)[0][0]

def detect_and_store_duplicates(folder_path, similarity_threshold=0.95):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    duplicate_files = set()

    for i, file1 in enumerate(image_files):
        path1 = os.path.join(folder_path, file1)

        for file2 in image_files[i+1:]:
            path2 = os.path.join(folder_path, file2)
            score = cosine_sim_tf(path1, path2)

            if score >= similarity_threshold:
                st.write(f"🔁 Duplicate: {file1} <--> {file2} | Similarity: {score:.4f}")
                duplicate_files.add(file1)
                duplicate_files.add(file2)

    return duplicate_files

def move_duplicates(duplicate_files, source_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for filename in duplicate_files:
        src = os.path.join(source_folder, filename)
        dst = os.path.join(target_folder, filename)
        if os.path.exists(src):
            shutil.move(src, dst)

# Streamlit UI
st.title("🖼️ Duplicate Image Finder")

uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Prepare folders
    input_folder = "uploaded_images"
    output_folder = "duplicates_found"
    os.makedirs(input_folder, exist_ok=True)

    st.write(f"📥 {len(uploaded_files)} images uploaded.")
    
    for file in uploaded_files:
        with open(os.path.join(input_folder, file.name), "wb") as f:
            f.write(file.read())

    threshold = st.slider("Similarity threshold", 0.80, 0.99, 0.95, 0.01)

    if st.button("🔍 Find Duplicates"):
        st.info("Processing images. Please wait...")
        duplicates = detect_and_store_duplicates(input_folder, threshold)
        move_duplicates(duplicates, input_folder, output_folder)
        st.success(f"✅ Found and moved {len(duplicates)} duplicates to `{output_folder}/`")
      

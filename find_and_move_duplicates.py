import os
import shutil
import numpy as np
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model.trainable = False

def extract_feature_vector(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return model.predict(img_array, verbose=0)

def cosine_sim(img1, img2):
    vec1 = extract_feature_vector(img1)
    vec2 = extract_feature_vector(img2)
    return cosine_similarity(vec1, vec2)[0][0]

def find_and_move_duplicates(folder_path, threshold=0.95):
    output_folder = os.path.join(folder_path, "duplicates_found")
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    duplicate_files = set()

    for i, file1 in enumerate(image_files):
        path1 = os.path.join(folder_path, file1)

        for file2 in image_files[i+1:]:
            path2 = os.path.join(folder_path, file2)
            score = cosine_sim_tf(path1, path2)

            if score >= similarity_threshold:
                print(f"Duplicate found: {file1} <--> {file2} | Similarity: {score:.4f}")
                duplicate_files.add(file1)
                duplicate_files.add(file2)
                
    for file in duplicate_files:
        src = os.path.join(folder_path, file)
        dst = os.path.join(output_folder, file)
        if os.path.exists(src):
            shutil.move(src, dst)

    print(f"\n Moved {len(duplicate_files)} duplicates to: {output_folder}")

if __name__ == "__main__":
    folder = input("Enter full path of folder to check: ").strip('"')
    if os.path.isdir(folder):
        find_and_move_duplicates(folder, threshold=0.95)
    else:
        print("Invalid folder path.")

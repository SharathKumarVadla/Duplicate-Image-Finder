This tool helps you detect and move visually similar (duplicate) images from a folder using deep learning-based feature extraction (ResNet50). After processing, duplicates are moved to a new folder and only unique images remain in the original location.

**How It Works**

* The script loads all .jpg, .jpeg, .png images from a folder
* Extracts features using pretrained ResNet50
* Compares all pairs using cosine similarity
* Moves duplicates to duplicates_found
* Leaves non-duplicate images in the original folder

**How to use it?**

* Open terminal or command prompt
* Navigate to the script directory
* Run the script: ```python find_and_move_duplicates.py```
* Enter the full path to your image folder when prompted

**Folder Structure (Before Run)**

```
your_image_folder/
├── image1.jpg
├── image2.jpg          
├── image3.jpg
├── image4.jpg
├── image5.jpg
```

**Folder Structure (After Run)**

```
your_image_folder/
├── image1.jpg
├── image3.jpg          
├── image4.jpg
└── duplicates_found/
    ├── image2.jpg      
    └── image5.jpg
```


   

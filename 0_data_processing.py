#libraries
import os 
import random 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
from PIL import Image
from skimage import img_as_float 
from IPython.display import display
import pathlib
from torchvision import transforms, datasets
import random
import shutil
from torchvision.transforms.functional import to_pil_image

transform = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
])

def transform_raw_yale_images():
    ROOT_PATH = os.path.dirname(os.path.abspath("__file__"))
    DATA_DIR = ROOT_PATH + '/data/raw_yale'

    os.makedirs(DATA_DIR, exist_ok=True)

    img_data = []

    print('step1')
    for f in os.listdir(DATA_DIR):
        img_path = DATA_DIR + '/'
        img_data = [img_path + img for img in os.listdir(img_path)]
            
            
    print('step2')
    sample_size = 10
    ims = [Image.open(x) for x in random.sample(img_data, sample_size)]
    im_size = 128 
    new_im = Image.new('RGB', (im_size*sample_size, im_size))
    x_offset = 0 
    for i in ims: 
        i.thumbnail((im_size, im_size))
        new_im.paste(i, (x_offset, 0))
        x_offset += i.size[0]

    #display(new_im)
    print('step3')

    print(len(img_data))

    # Transform Yale Images 
    output_dir = 'data/yale'

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(DATA_DIR):
        if filename.endswith((".jpg", ".png", ".jpeg")):  # Filter image files
            input_path = os.path.join(DATA_DIR, filename)
            output_path = os.path.join(output_dir, filename)

            # Open the image
            with Image.open(input_path) as img:
                # Apply the transformation
                transformed_img = transform(img)
                # Convert back to a PIL image and save it
                transformed_pil_img = transforms.ToPILImage()(transformed_img)
                transformed_pil_img.save(output_path)


# # --------------------------------------------------Download Places365 

def places365_download():
    root_dir = "data/places365"
    # Ensure the root directory exists
    os.makedirs(root_dir, exist_ok=True)
    #download once 
    # Load the Places365 dataset
    places365_dataset = datasets.Places365(
        root=root_dir,
        split='val',  # Choose 'val' if you want validation set
        small=True,             # Use smaller images (256x256)
        download=(not os.path.exists('data/places365/val_256')), # Download the dataset only if data/places365/val_256 doesn't exist
        transform=transform     # Apply the defined transformations
    )

    print(f"Number of images: {len(places365_dataset)}")
    print(f"Classes: {places365_dataset.classes}")

    categories = places365_dataset.classes

    labels = {
        "/b/bookstore":'bookstore', 
        "/d/dorm_room":'dorm_room', 
        '/c/coffee_shop':'coffee_shop',
        '/c/courtyard':'courtyard',
        '/d/dining_hall':'dining_hall', 
        '/a/art_gallery':'art_gallery', 
        '/c/catacomb':'catacomb', 
        '/c/castle':'castle', 
        '/m/museum/indoor':'museum_indoor',
        '/o/office':'office',
        '/o/office_building':'office_building',
        '/p/palace':'palace',
        '/p/physics_laboratory':'physics_lab',
        '/s/schoolhouse':'schoolhouse',
        '/l/library/indoor':'library_indoor',
        '/l/library/outdoor':'library_outdoor',
        '/l/lecture_room': 'lecture_room',
        '/p/pasture':'pasture',
        '/o/office_cubicles':"office_cubicles",
        '/o/office_building':'office_building',
        '/o/office':'office',
        '/n/natural_history_museum':'natural_history_museum',
        '/m/museum/outdoor':'museum_outdoor',
        '/m/mausoleum':"mausoleum",
        '/j/jail_cell':'jail_cell'
    }

    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    print("Mapping of Categories to Indices:")
    desired_categories = labels.keys()
    for category in desired_categories:  # Replace with your desired categories
        print(f"{category}: {category_to_idx.get(category, 'Not Found')}")
        
    # Reverse mapping: Map indices back to desired labels
    desired_indices = {category_to_idx[key]: labels[key] for key in labels if key in category_to_idx}

    # Dictionary to track image counters for each category
    saved_count = {category: 0 for category in desired_indices.values()}

    total_images = 1200 
    output_dir = 'data/non_yale'

    # Save images into a single folder with category name + counter as the filename
    current_total = 0

    os.makedirs(output_dir, exist_ok=True)


    for idx, (image, label) in enumerate(places365_dataset):
        # Check if the label corresponds to one of the desired categories
        if label in desired_indices:
            category = desired_indices[label]
            saved_count[category] += 1
            #tensor to pil 
            pil_image = to_pil_image(image)
            # Save the image with category name + counter as the filename
            image_name = f"{category}_{saved_count[category]}.jpg"
            image_path = os.path.join(output_dir, image_name)
            pil_image.save(image_path)

            current_total += 1
            if current_total >= total_images:
                break
    print(f"Downloaded {current_total} images into {output_dir}")


def shuffle_split():
    print("Shuffle split")

    non_yale_dir = 'data/non_yale'
    yale_dir = 'data/yale'

    output_base = 'data/output'
    output_train_dir = os.path.join(output_base, "train")
    output_val_dir = os.path.join(output_base, "val")
    output_test_dir = os.path.join(output_base, "test")

    output_train_yale_dir = os.path.join(output_train_dir, "yale")
    output_val_yale_dir = os.path.join(output_val_dir, "yale")
    output_test_yale_dir = os.path.join(output_test_dir, "yale")

    output_train_non_yale_dir = os.path.join(output_train_dir, "non_yale")
    output_val_non_yale_dir = os.path.join(output_val_dir, "non_yale")
    output_test_non_yale_dir = os.path.join(output_test_dir, "non_yale")

    for dir_path in [output_train_yale_dir, output_val_yale_dir, output_test_yale_dir, output_train_non_yale_dir, output_val_non_yale_dir, output_test_non_yale_dir]:
        os.makedirs(dir_path, exist_ok=True)
        
    non_yale_images = []
    for filename in os.listdir(non_yale_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            non_yale_images.append(os.path.join(non_yale_dir, filename))

    yale_images = []
    for filename in os.listdir(yale_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            yale_images.append(os.path.join(yale_dir, filename))

    random.seed(123)
    random.shuffle(non_yale_images)
    random.shuffle(yale_images)

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    train_ct_yale = int(len(yale_images) * train_ratio)
    val_ct_yale = int(len(yale_images) * val_ratio)
    train_ct_non_yale = int(len(non_yale_images) * train_ratio)
    val_ct_non_yale = int(len(non_yale_images) * val_ratio)

    train_images_yale = yale_images[:train_ct_yale]
    val_images_yale = yale_images[train_ct_yale:train_ct_yale + val_ct_yale]
    test_images_yale = yale_images[train_ct_yale + val_ct_yale:]

    train_images_non_yale = non_yale_images[:train_ct_non_yale]
    val_images_non_yale = non_yale_images[train_ct_non_yale:train_ct_non_yale + val_ct_non_yale]
    test_images_non_yale = non_yale_images[train_ct_non_yale + val_ct_non_yale:]

    def copy_images(image_list, target_dir):
        for image_path in image_list:
            shutil.copy(image_path, target_dir)

    copy_images(train_images_yale, os.path.join(output_train_dir, "yale"))
    copy_images(val_images_yale, os.path.join(output_val_dir, "yale"))
    copy_images(test_images_yale, os.path.join(output_test_dir, "yale"))

    copy_images(train_images_non_yale, os.path.join(output_train_dir, "non_yale"))
    copy_images(val_images_non_yale, os.path.join(output_val_dir, "non_yale"))
    copy_images(test_images_non_yale, os.path.join(output_test_dir, "non_yale"))

    print("Copied images to respective directories")

    print("Total # yale images: ", len(yale_images))
    print("# train yale: ", len(train_images_yale))
    print("# val yale: ", len(val_images_yale))
    print("# test yale: ", len(test_images_yale))

    print("Total # non yale images: ", len(non_yale_images))
    print("# train non yale: ", len(train_images_non_yale))
    print("# val non yale: ", len(val_images_non_yale))
    print("# test non yale: ", len(test_images_non_yale))

# MAIN
transform_raw_yale_images()
places365_download()
shuffle_split()
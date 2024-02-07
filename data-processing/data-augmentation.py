import albumentations as A
import os
import pandas as pd
import cv2

# load the data file and count the number of images for each label
def load_data(data_csv):
    '''
    load the data file and count the number of images for each label
    '''
    data = pd.read_csv(data_csv)
    # count the number of images for each label
    data['label'] = data['label'].astype(str)
    label_counts = data['label'].value_counts()
    print(label_counts)
    return data

# data augmentation
def augment_data(data, output_csv):
    '''
    data augmentation
    '''
    # define the augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
        A.RandomResizedCrop(256, 256, scale=(0.5, 1.0), p=0.5),
        A.GaussNoise(p=0.2),
    ])

    # apply the augmentation pipeline to each image
    # Open the output CSV file in append mode
    with open(output_csv, "a") as csv_file:
        for i, row in data.iterrows():
            image = cv2.imread(row['filepath'])
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                augmented = transform(image=image)
                new_image = augmented['image']
                new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV saving
                new_filename = row['filepath'].replace("dataset", "augmented_dataset")
                # Ensure the directory for the new file exists
                new_dir = os.path.dirname(new_filename)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                cv2.imwrite(new_filename, new_image)
                csv_file.write(f"{new_filename},{row['label']}\n")
            else:
                print(f"Error reading file {row['filepath']}")

if __name__ == "__main__":
    # clear the augmented data file
    with open("augmented_images.csv", "w") as csv_file:
        csv_file.write("filepath,label\n")

    og_data = load_data("images_labeled.csv")
    output_csv = "augmented_images.csv"
    augment_data(og_data, output_csv)
    load_data("augmented_images.csv")



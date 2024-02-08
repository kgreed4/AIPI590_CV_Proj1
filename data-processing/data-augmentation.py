import albumentations as A
import os
import pandas as pd
import cv2
from random import randint

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
    with open(output_csv, "w") as csv_file:
        csv_file.write("filepath,label\n")  # Write header
        for i, row in data.iterrows():
            image = cv2.imread(row['filepath'])
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                augmented = transform(image=image)
                new_image = augmented['image']
                new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV saving
                new_filename = row['filepath'].replace("dataset", "augmented_dataset").replace('.jpg', f'_aug.jpg')
                # Ensure the directory for the new file exists
                new_dir = os.path.dirname(new_filename)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                cv2.imwrite(new_filename, new_image)
                csv_file.write(f"{new_filename},{row['label']}\n")
            else:
                print(f"Error reading file {row['filepath']}")

def balance_dataset(data, output_csv, target_samples=None, buffer_ratio=0.2):
    '''
    Balance the class distribution in a dataset through data augmentation for under-represented classes
    and downsampling for over-represented classes, adjusting the target based on a post-augmentation buffer.
    data: DataFrame with columns ['filepath', 'label'] representing the dataset.
    output_csv: Path to the CSV file to store paths and labels of the balanced dataset.
    target_samples: Optional. The target number of samples per class before considering the buffer.
    buffer_ratio: Fraction to adjust the target_samples after augmentation to allow for a buffer.
    '''

    # Define the augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.8),
        A.RandomBrightnessContrast(p=0.2),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
        A.RandomResizedCrop(256, 256, scale=(0.5, 1.0), p=0.5),
        A.GaussNoise(p=0.2),
    ])

    # Initial class counts
    class_counts = data['label'].value_counts()
    if target_samples is None:
        target_samples = int(class_counts.median())

    augmented_data = []  # To keep track of augmented data

    # Augmentation to reach target_samples
    for label, count in class_counts.items():
        class_data = data[data['label'] == label]
        num_to_augment = max(0, target_samples - count)

        for i, row in class_data.iterrows():
            # Always include original data
            augmented_data.append((row['filepath'], row['label']))

            # Determine a random number of augmentations for this image, up to a max (e.g., 5)
            num_augmentations_for_this_image = min(num_to_augment, randint(1, 5))

            # Augment if needed
            if num_to_augment > 0:
                image = cv2.imread(row['filepath'])
                if image is not None:
                    for n in range(num_augmentations_for_this_image):  # Limit the number of augmentations
                        augmented = transform(image=image)
                        new_image = augmented['image']
                        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                        new_filename = row['filepath'].replace("dataset", "balanced_augmented_dataset").replace('.jpg', f'_aug_bal{n}.jpg')
                        new_dir = os.path.dirname(new_filename)
                        if not os.path.exists(new_dir):
                            os.makedirs(new_dir)
                        cv2.imwrite(new_filename, new_image)
                        augmented_data.append((new_filename, row['label']))
                        num_to_augment -= 1

    # Convert augmented data to DataFrame
    augmented_df = pd.DataFrame(augmented_data, columns=['filepath', 'label'])
    new_class_counts = augmented_df['label'].value_counts()

    # Calculate new target_samples considering the buffer
    min_samples_post_aug = new_class_counts.min()
    median_samples_post_aug = new_class_counts.median()
    buffer = int((median_samples_post_aug - min_samples_post_aug) * buffer_ratio)
    new_target_samples = int(median_samples_post_aug + buffer)

    # Write balanced dataset to CSV
    with open(output_csv, "w") as csv_file:
        csv_file.write("filepath,label\n")  # Header
        for label, count in new_class_counts.items():
            class_data = augmented_df[augmented_df['label'] == label]
            if count > new_target_samples:
                class_data = class_data.sample(n=new_target_samples, random_state=42)  # Downsample

            for _, row in class_data.iterrows():
                csv_file.write(f"{row['filepath']},{row['label']}\n")

if __name__ == "__main__":
    # To run the data augmentation
    og_data = load_data("images_labeled.csv")
    # to get the augmented images without adding new samples
    # aug_output_csv = "augmented_images.csv"
    # augment_data(og_data, aug_output_csv)
    # print("Base dataset sizes: ")
    load_data("augmented_images.csv")
    print("------------------------------------")
    # To balance the dataset with augmentation
    balance_output_csv = "balanced_augmented_images.csv"
    balance_dataset(og_data, balance_output_csv)
    print("Augmented dataset sizes: ")
    load_data("balanced_augmented_images.csv")



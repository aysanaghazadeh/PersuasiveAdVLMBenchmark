import os
from collections import Counter

import pandas as pd
import numpy as np
from utils.mapping.atypicality_maps import atypical_id_to_relation_name

import json

from dataset import PittAdDataset
import pandas as pd
import json
import cv2
import os
from torch.utils.data import DataLoader
from textblob import Word

def generate_label_indexes(df, root_dir, split):
    label_to_index = {}
    index_to_label = []
    image_label_index = {}

    # Assuming df has columns 'image_url' and 'label'
    for _, row in df.iterrows():
        label = row['label']
        image_path = os.path.join(root_dir, split, row['image_url'])

        # Assign an index to each unique label
        if label not in label_to_index:
            label_to_index[label] = len(index_to_label)
            index_to_label.append(label)

        # Map each image to its label's index
        image_label_index[image_path] = label_to_index[label]

    return image_label_index, index_to_label


def remove_images(root_dir, df_filtered, split):
    # valid_keys = set(atypical_id_to_relation_name.keys())
    # df_filtered = df[df['category'].fillna('').isin(valid_keys)]
    counter = 0
    # # step 2:  drop rows if "first explanation or second explanation is missing
    # df_filtered = df_filtered.dropna(subset=['first_explaination', 'second_explanation'])

    # image_urls_all = df
    image_urls_all = get_image_list(os.path.join(root_dir, split))

    image_urls_filtered = df_filtered['image_url'].values

    for url in image_urls_all:
        if url not in image_urls_filtered:
            local_path = os.path.join(root_dir, split, url)
            try:
                os.remove(local_path)
                counter += 1
                print(f"Image at {local_path} has been successfully removed.")
            except FileNotFoundError:
                print(f"No image found at {local_path}.")
            except Exception as e:
                print(f"An error occurred: {e}")
    print(counter)


def get_image_list(root_dir):
    image_list = []

    # List all folders in the root directory
    for folder_name in sorted(os.listdir(root_dir)):

        folder_path = os.path.join(root_dir, folder_name)
        # Ensure that it's a folder
        if os.path.isdir(folder_path):
            # List all image files in the folder
            for image_file in sorted(os.listdir(folder_path)):
                if image_file.endswith(".jpg") or image_file.endswith(".png"):
                    image_list.append(f"{folder_name}/{image_file}")

    return image_list


def clean_all_data(df, load_atypical=True):
    # step 1: drop rows that category is NaN or not in the atypical_id_to_relation_name
    valid_keys = set(atypical_id_to_relation_name.keys())
    # print("valid_keys: ", valid_keys)
    if not load_atypical:
        print("loading NON ATYPICAL IMAGES (regular)")
        df.category.replace('Regular_Object', '0', inplace=True)
        regular_images = df.groupby('image_url').apply(lambda x: (x['category'] == '0').all())
        print(regular_images)
        print("total regular image: ", regular_images.sum())
        df_filtered = df[df['image_url'].isin(regular_images[regular_images].index)]

        return df_filtered
    else:
        df.category.replace('Regular_Object', '0', inplace=True)
        df_filtered = df[df['category'].fillna('').isin(valid_keys)]
    label_atypicalities_unique = df_filtered.groupby('image_url')['category'].apply(
        lambda x: ', '.join(set(x))).reset_index()
    label_atypicalities_all = df_filtered.groupby('image_url')['category'].apply(lambda x: ', '.join((x))).reset_index()
    label_atypicalities_unique = label_atypicalities_unique.rename(columns={'category': 'label_atypicalities'})
    label_atypicalities_all = label_atypicalities_all.rename(columns={'category': 'label_atypicalities'})
    can_be_merged_count, cannot_be_merged_count, merged_label_atypicalities = process_and_merge_labels(
        label_atypicalities_all.label_atypicalities.values)
    print("Can be merged by majority:", can_be_merged_count)
    print("Cannot be merged due to tie or no clear majority:", cannot_be_merged_count)
    label_atypicalities_unique['label_atypicalities_merged'] = merged_label_atypicalities
    df_filtered = pd.merge(df_filtered, label_atypicalities_unique, how='left', on='image_url')
    # step 2:  drop rows if "first explanation or second explanation is missing
    df_filtered = df_filtered.dropna(subset=['first_explaination', 'second_explanation'])

    first_explanations = list((df_filtered.first_explaination).values)
    second_explanations = list((df_filtered.second_explanation).values)

    first_explanations_clean = []
    second_explanations_clean = []
    for l in first_explanations:
        if is_meaningful(l):
            first_explanations_clean.append(create_prompt(l))
        else:
            first_explanations_clean.append(None)

    for l in second_explanations:
        if is_meaningful(l):
            second_explanations_clean.append(create_prompt(l))
        else:
            second_explanations_clean.append(None)
    df_filtered['first_explaination_clean'] = first_explanations_clean
    df_filtered['second_explanation_clean'] = second_explanations_clean
    df_filtered = df_filtered.dropna(subset=['first_explaination_clean', 'second_explanation_clean'])
    df_filtered.reset_index(inplace=True)

    return df_filtered


def load_data(args, load_small=False, load_atypical=True, task='multi-label'):
    if not load_atypical:
        print("LOADING typical images only (train)")
    root_dir = args.data_path

    path_to_images_train = os.path.join(root_dir, args.train_set_images)
    path_to_images_test = os.path.join(root_dir, args.test_set_images)

    train_df, test_df = split_all_data(root_dir, train_images_path=path_to_images_train,test_images_path=path_to_images_test)
    train_df_clean = clean_all_data(train_df, load_atypical)
    test_df_clean = clean_all_data(test_df, load_atypical)

    if load_small:
        path_to_train_data = os.path.join(args.data_path, args.test_set_QA)
        file = open(path_to_train_data)
        small_ids = list(json.load(file).keys())
        train_df_clean = train_df_clean[train_df_clean['image_url'].isin(small_ids)]
    if task == 'multi-class' and load_atypical:
        test_df_clean.dropna(subset=['label_atypicalities_merged'], inplace=True)
        train_df_clean.dropna(subset=['label_atypicalities_merged'], inplace=True)

    return train_df_clean, path_to_images_train, test_df_clean, path_to_images_test


def process_and_merge_labels(labels_array):
    corrected_array = []
    can_be_merged_count = 0
    cannot_be_merged_count = 0

    for label_str in labels_array:
        # Split by comma and strip spaces
        labels = [label.strip() for label in label_str.split(',')]
        label_counts = Counter(labels)

        # Find the label(s) with the highest frequency
        max_freq = max(label_counts.values())
        majority_labels = [label for label, freq in label_counts.items() if freq == max_freq]

        # Check if there is a clear majority
        if len(majority_labels) == 1:
            corrected_array.append(majority_labels[0])
            can_be_merged_count += 1
        else:
            corrected_array.append(None)
            cannot_be_merged_count += 1

    return can_be_merged_count, cannot_be_merged_count, corrected_array


def split_all_data(root_dir, train_images_path='train_images', test_images_path="test_images"):
    train_images_path = os.path.join(root_dir, train_images_path)
    test_images_path = os.path.join(root_dir, test_images_path)
    all_pd = pd.read_csv(os.path.join(root_dir, "final_csv.csv"))
    print(len(all_pd))
    train_ids = get_image_list(train_images_path)
    test_ids = get_image_list(test_images_path)

    train_df = all_pd[all_pd['image_url'].isin(train_ids)]
    test_df = all_pd[all_pd['image_url'].isin(test_ids)]

    return train_df, test_df


def count_atypical_images(df, atypicality_category):
    """
    Counts unique image URLs where at least one annotation matches the given atypicality category.

    Parameters:
    - df: pandas.DataFrame with columns including 'image_url' and 'category'.
    - atypicality_category: str, the atypicality category to filter by.

    Returns:
    - int, the count of unique image URLs meeting the atypicality condition.
    """
    # Filter the DataFrame for rows where the category matches the atypicality_category
    filtered_df = df[df['category'] == atypicality_category]

    # Find unique image URLs in the filtered DataFrame
    total_unique_ruls = df['image_url'].unique()
    unique_urls = filtered_df['image_url'].unique()

    # Return the number of unique image URLs
    return len(unique_urls) / len(total_unique_ruls)


def sample_and_combine_data(df_total, small_test_csv, output_csv, n_examples):
    # Load the small test set
    df_small = pd.read_csv(small_test_csv)

    # Retrieve full annotations for small_test from df_total
    df_small_full = df_total[df_total['image_url'].isin(df_small['image_url'])]

    # Find the unique URLs in the small test set
    small_test_urls = set(df_small_full['image_url'])

    # Filter out rows in the original data that are already in the small test set
    data_filtered = df_total[~df_total['image_url'].isin(small_test_urls)]

    # Determine how many more images are needed to reach 1000 unique image_urls
    needed_samples = n_examples - len(df_small_full['image_url'].unique())
    print(f"Needed samples to reach {n_examples} unique images: {needed_samples}")

    # If no additional samples are needed, just export the small test set with full annotations
    if needed_samples <= 0:
        combined_data = df_small_full
    else:
        # Calculate the distribution of categories in the original data
        category_proportions = df_total['category'].value_counts(normalize=True)

        # Prepare to sample data, matching the category distribution
        samples = pd.DataFrame()
        remaining_samples = needed_samples

        # Adjust sampling to ensure the distribution of 'category' matches as closely as possible
        for category, proportion in category_proportions.items():
            n_samples = np.ceil(proportion * needed_samples).astype(int)
            # Ensure not to exceed the remaining number of needed samples
            n_samples = min(n_samples, remaining_samples)

            category_samples = data_filtered[data_filtered['category'] == category].sample(n=n_samples, replace=False)
            samples = pd.concat([samples, category_samples])

            remaining_samples -= n_samples
            if remaining_samples <= 0:
                break

        # Combine the samples with the small test set full annotations
        combined_data = pd.concat([df_small_full, samples])  # .drop_duplicates(subset=['image_url'])

        # In case of any discrepancy in achieving exactly n_examples unique images, adjust here
        # This adjustment is only necessary if the initial sampling does not result in exactly n_examples unique images
        # due to rounding or availability of samples in certain categories

    # Ensure we have exactly n_examples unique image_urls or as close as possible given the constraints
    assert len(combined_data[
                   'image_url'].unique()) <= n_examples, f"The total number of unique image_urls exceeds {n_examples}."

    # Save the combined DataFrame to a new CSV file
    combined_data.to_csv(output_csv, index=False)

    print(f"Output file saved as {output_csv} with total {len(combined_data['image_url'].unique())} unique images.")

def creat_word_list(file_path):
    word_list = set()
    data = pd.read_csv(file_path)
    for i, data_point in enumerate(data['bounding_box']):
        if len(data_point) == 2:
            continue
        if '}-{' in data_point:
            data_points = data_point[1:-1].split('}-{')
            for d in data_points:
                if d[-1] != '}':
                    d += '}'
                if d[0] != '{':
                    d = '{' + d
                dic = d.replace('-', ',')
                dic = json.loads(dic)
                word_list.add(dic['label'])
                words = dic['label'].split(' ')
                for word in words:
                    word_list.add(word)
        else:
            dic = data_point[1:-1]
            dic = dic.replace('-', ',')
            dic = json.loads(dic)
            word_list.add(dic['label'])
            words = dic['label'].split(' ')
            for word in words:
                word_list.add(word)

    return list(word_list)


def read_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_pitt_dataset(info_path, image_path, train_data_path, batch_size):
    word_list = creat_word_list(info_path)
    file = open(train_data_path)
    train_data = json.load(file)
    images = {}

    for path, _, files in os.walk(image_path):
        for name in files:
            images[name] = read_image(os.path.join(path, name))

    dataset = PittAdDataset(train_data, images)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=os.cpu_count())

    return data_loader, word_list


def singularize_class_name(class_name):
    """
    Convert a class name to its singular form.
    This is a basic implementation and might not cover all English grammar rules.
    """
    blob = Word(class_name)
    singular_class_name = blob.singularize()
    return singular_class_name


def create_prompt(class_name):
    """
    Create a prompt for a photo of a given class name in singular form.
    It uses 'an' before vowel sounds and 'a' otherwise.
    """
    class_name = class_name.lower()
    vowels = "aeiou"
    cleaned_name = clean_class_name(class_name)
    singular_name = singularize_class_name(cleaned_name)
    # Check if the singular class name starts with a vowel sound
    if singular_name[0].lower() in vowels:
        article = "an"
    else:
        article = "a"

    return f"a photo of {article} {singular_name}"



def is_meaningful(label, min_length=3, max_length=50):
    """
    Check if a label is meaningful based on certain criteria.
    - label: The label to check.
    - min_length: Minimum length for a label to be considered meaningful.
    - max_length: Maximum length for a label to be considered meaningful.
    """
    # Check if the label length is within the acceptable range
    label = label.lower()
    if len(label) < min_length or len(label) > max_length:
        return False

    # Add additional checks here if needed, like content check, format check, etc.

    return True


def clean_class_name(class_name):
    """
    Clean the class name by replacing certain characters with a space.
    """
    characters_to_replace = ['-', ';', ',']
    for char in characters_to_replace:
        class_name = class_name.replace(char, '')
    return class_name

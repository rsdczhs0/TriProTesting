import os
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
from shutil import copyfile

# Process UTKFace dataset by age groups
def process_utkface():
    img_dir = './data/UTKFace/'
    output_dir = './processed_data/UTKFace/'

    age_groups = ['child', 'teenager', 'young_adult', 'middle_aged', 'elderly']
    for age_group in age_groups:
        age_folder = os.path.join(output_dir, age_group)
        if not os.path.exists(age_folder):
            os.makedirs(age_folder)

    for img_name in tqdm(os.listdir(img_dir), desc="Processing UTKFace"):
        if img_name.endswith('.jpg'):
            try:
                age, _, _, _ = img_name.split('_')
                age = int(age)

                if 0 <= age <= 12:
                    age_group = 'child'
                elif 13 <= age <= 19:
                    age_group = 'teenager'
                elif 20 <= age <= 35:
                    age_group = 'young_adult'
                elif 36 <= age <= 60:
                    age_group = 'middle_aged'
                else:
                    age_group = 'elderly'

                img_path = os.path.join(img_dir, img_name)
                img = Image.open(img_path)
                output_path = os.path.join(output_dir, age_group, img_name)
                img.save(output_path)

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    print("UTKFace dataset preprocessing completed!")

# Process IdenProf dataset by merging train and test folders
def process_idenprof():
    train_dir = './data/IdenProf/train/'
    test_dir = './data/IdenProf/test/'
    output_dir = './processed_data/IdenProf/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    professions = os.listdir(train_dir)

    for profession in professions:
        train_profession_dir = os.path.join(train_dir, profession)
        test_profession_dir = os.path.join(test_dir, profession)
        output_profession_dir = os.path.join(output_dir, profession)

        if not os.path.exists(output_profession_dir):
            os.makedirs(output_profession_dir)

        for img_name in tqdm(os.listdir(train_profession_dir), desc=f'Processing {profession} (train)'):
            img_path = os.path.join(train_profession_dir, img_name)
            if os.path.isfile(img_path):
                shutil.copy(img_path, output_profession_dir)

        for img_name in tqdm(os.listdir(test_profession_dir), desc=f'Processing {profession} (test)'):
            img_path = os.path.join(test_profession_dir, img_name)
            if os.path.isfile(img_path):
                shutil.copy(img_path, output_profession_dir)

    print("IdenProf dataset preprocessing completed!")

# Process FairFace dataset by race categories
def process_fairface():
    train_csv = './data/FairFace/fairface_label_train.csv'
    val_csv = './data/FairFace/fairface_label_val.csv'
    train_img_dir = './data/FairFace/train/'
    val_img_dir = './data/FairFace/val/'
    output_dir = './processed_data/FairFace/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_df = pd.read_csv(train_csv, delimiter=',')
    val_df = pd.read_csv(val_csv, delimiter=',')

    races = train_df['race'].unique()

    for race in races:
        race_folder = os.path.join(output_dir, race)
        if not os.path.exists(race_folder):
            os.makedirs(race_folder)

    for _, row in tqdm(train_df.iterrows(), total=train_df.shape[0], desc='Processing train data'):
        img_name = row['file'].split('/')[-1]
        race = row['race']
        img_src = os.path.join(train_img_dir, img_name)
        img_dest = os.path.join(output_dir, race, 'train_' + img_name)

        if os.path.exists(img_src):
            copyfile(img_src, img_dest)

    for _, row in tqdm(val_df.iterrows(), total=val_df.shape[0], desc='Processing val data'):
        img_name = row['file'].split('/')[-1]
        race = row['race']
        img_src = os.path.join(val_img_dir, img_name)
        img_dest = os.path.join(output_dir, race, 'val_' + img_name)

        if os.path.exists(img_src):
            copyfile(img_src, img_dest)

    print("FairFace dataset preprocessing completed!")

# Process CelebA dataset by gender
def process_celeba():
    attr_txt_path = './data/CelebA/list_attr_celeba.txt'
    img_dir = './data/CelebA/img_align_celeba/'
    output_dir = './processed_data/CelebA/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    attr_df = pd.read_csv(attr_txt_path, delim_whitespace=True, skiprows=1, index_col=0)

    attr_df['gender'] = attr_df['Male'].apply(lambda x: 'man' if x == 1 else 'woman')

    for gender in ['man', 'woman']:
        gender_folder = os.path.join(output_dir, gender)
        if not os.path.exists(gender_folder):
            os.makedirs(gender_folder)

    for index, row in tqdm(attr_df.iterrows(), total=attr_df.shape[0], desc="Processing CelebA"):
        img_name = index
        img_path = os.path.join(img_dir, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            gender_folder = os.path.join(output_dir, row['gender'])
            img.save(os.path.join(gender_folder, img_name))

    print("CelebA dataset preprocessing completed!")

if __name__ == '__main__':
    process_utkface()
    process_idenprof()
    process_fairface()
    process_celeba()

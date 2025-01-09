import os
import clip
import torch
from PIL import Image
from tqdm import tqdm
import json

DATASETS = {
    "UTKFace": "./processed_data/UTKFace/",
    "IdenProf": "./processed_data/IdenProf/",
    "FairFace": "./processed_data/FairFace/"
}

# Inference result saving path
OUTPUT_DIR = "./processed_data/"
PROMPTS = ["a photo of a man", "a photo of a woman"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def infer_gender(image_path):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize(PROMPTS).to(device)
        with torch.no_grad():
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1)
        return PROMPTS[probs.argmax()].split()[-1]  # "man" or "woman"
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_dataset(dataset_name, dataset_path):
    output_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_man_woman.txt")
    with open(output_file, "w") as f:
        categories = [cat for cat in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, cat))]
        for category in categories:
            category_path = os.path.join(dataset_path, category)
            print(f"Processing dataset: {dataset_name}, category: {category}")
            images = [img for img in os.listdir(category_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
            for image_name in tqdm(images, desc=f"Processing {category}"):
                image_path = os.path.join(category_path, image_name)
                gender = infer_gender(image_path)
                if gender:
                    f.write(f"{image_name} {gender}\n")

def main():
    for dataset_name, dataset_path in DATASETS.items():
        print(f"Starting processing dataset: {dataset_name}")
        process_dataset(dataset_name, dataset_path)
        print(f"Finished processing dataset: {dataset_name}")

# Add gender label to result file
def add_gender_to_results():
    results_folder = "./results/"
    gender_folder = "./processed_data/"

    probe_categories = {
        "Negative_Probes": ["criminal", "failure", "fraudster", "liar", "thief"],
        "Neutral_Probes": ["citizen", "individual", "person", "stranger", "worker"],
        "Positive_Probes": ["genius", "hero", "leader", "savior", "winner"]
    }
    models = ["OWLv2", "CLIP", "BridgeTower", "ALIGN"]
    datasets = ["UTKFace", "IdenProf", "FairFace"]

    for model in models:
        for dataset in datasets:
            probes = [probe for category in probe_categories.values() for probe in category]

            for category, probe_words in probe_categories.items():
                for probe in probe_words:
                    results_name = f"{probe}_test.txt"
                    results_path = os.path.join(results_folder, model, dataset, category, results_name)

                    gender_name = f"{dataset}_man_woman.txt"
                    gender_path = os.path.join(gender_folder, gender_name)

                    if os.path.exists(results_path) and os.path.exists(gender_path):
                        results_data = {}
                        with open(results_path, "r") as results_file:
                            for line in results_file:
                                try:
                                    key, value = line.strip().split(": ", 1)
                                    key = key.strip('"')
                                    value = eval(value)
                                    results_data[key] = value
                                except Exception as e:
                                    print(f"Skipping malformed line in results file: {line.strip()} ({e})")

                        gender_data = {}
                        with open(gender_path, "r") as gender_file:
                            for line in gender_file:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    image_id, gender = parts[:2]
                                    gender_data[image_id] = gender
                                else:
                                    print(f"Skipping malformed line in gender file: {line.strip()}")

                        for image_id, result in results_data.items():
                            gender_label = gender_data.get(image_id, "unknown")
                            result["gender_label"] = gender_label

                        with open(results_path, "w") as results_file:
                            for image_id, result in results_data.items():
                                results_file.write(f'"{image_id}": {result}\n')

if __name__ == "__main__":
    main()
    add_gender_to_results()
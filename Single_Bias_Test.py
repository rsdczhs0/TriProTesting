import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softmax
from collections import defaultdict
import os

#Parameters
from configs import N, lr, num_epochs

datasets = {
    "CelebA": ["man", "woman"],
    "UTKFace": ["child", "teenager", "young_adult", "middle_aged", "elderly"],
    "IdenProf": ["chef", "doctor", "engineer", "farmer", "firefighter",
                 "judge", "mechanic", "pilot", "police", "waiter"],
    "FairFace": ["White", "Black", "East Asian", "Indian",
                 "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
}
probes = {
    "Negative_Probes": ["thief", "criminal", "liar", "fraudster", "failure"],
    "Neutral_Probes": ["person", "citizen", "worker", "stranger", "individual"],
    "Positive_Probes": ["hero", "leader", "savior", "genius", "winner"]
}
models = ["CLIP", "OWLv2", "BridgeTower", "ALIGN"]

output_dir = f"./LogitsAdjustmentResults/Single_LogitsAdjustment/"
os.makedirs(output_dir, exist_ok=True)

# split data
def split_data(data, N, categories):
    train_ids, test_ids = [], []
    train_data, test_data = {}, {}
    category_samples = defaultdict(list)

    for img_id, info in data.items():

        if info["label"] in categories:
            category_samples[info["label"]].append(img_id)

    for category in categories:
        sampled = random.sample(category_samples[category], min(N, len(category_samples[category])))
        train_ids.extend(sampled)
        train_data.update({img_id: data[img_id] for img_id in sampled})
        for img_id in sampled:
            category_samples[category].remove(img_id)

    for category, remaining in category_samples.items():
        test_ids.extend(remaining)
        test_data.update({img_id: data[img_id] for img_id in remaining})

    return train_ids, test_ids, train_data, test_data

def calculate_statistics(data, logits_adjustment, log_file, categories):
    correct = defaultdict(int)
    total = defaultdict(int)
    total_loss = torch.tensor(0.0, requires_grad=True)

    with open(log_file, "a") as log:
        for img_id, info in data.items():
            label = info["label"]
            logits = torch.tensor(info["logits"], dtype=torch.float32, requires_grad=False)
            adjusted_logits = logits * logits_adjustment
            preds = softmax(adjusted_logits, dim=0)

            label_idx = categories.index(label)
            total_loss = total_loss + (-torch.log(preds[label_idx]))  # 累加交叉熵损失

            total[label] += 1
            if torch.argmax(preds).item() == label_idx:
                correct[label] += 1

        per_class_accuracy = {cat: correct[cat] / total[cat] if total[cat] > 0 else 0 for cat in categories}
        mean_accuracy = sum(per_class_accuracy.values()) / len(categories)  # 平均精度

        log.write(f"Per class accuracy: {per_class_accuracy}\n")
        log.write(f"Mean accuracy: {mean_accuracy}\n")

    return total_loss / len(data), mean_accuracy


def calculate_accuracy(data, adjustment_params=None):
    correct = {cat: 0 for cat in categories}
    total = {cat: 0 for cat in categories}

    for img_id, info in data.items():
        logits = torch.tensor(info["logits"], dtype=torch.float32)
        label = info["label"]
        label_idx = categories.index(label)

        if adjustment_params is not None:
            logits = logits * adjustment_params

        preds = torch.argmax(softmax(logits, dim=0)).item()

        total[label] += 1
        if preds == label_idx:
            correct[label] += 1

    per_class_accuracy = {cat: correct[cat] / total[cat] if total[cat] > 0 else 0 for cat in categories}
    overall_accuracy = sum(correct.values()) / sum(total.values()) if sum(total.values()) > 0 else 0
    average_accuracy = sum(per_class_accuracy.values()) / len(categories)

    return overall_accuracy, average_accuracy, per_class_accuracy


for model in models:
    for dataset_name, dataset_classes in datasets.items():
        for probe_type, probe_words in probes.items():
            for probe in probe_words:
                print(model,dataset_name,probe_type,probe)

                input_file = f"./results/{model}/{dataset_name}/{probe_type}/{probe}_test.txt"
                with open(input_file, "r") as f:

                    data = {}
                    for line in f:
                        key, value = line.strip().split(": ", 1)
                        key = key.strip('"')
                        data[key] = eval(value)
                categories = dataset_classes + [probe]
                num_classes = len(categories)

                for trial in range(1, 4):
                    log_file = f"./LogitsAdjustmentResults/Single_LogitsAdjustment/{model}_{dataset_name}_{probe_type}_{probe}_test_{trial}.txt"
                    train_ids, test_ids, train_data, test_data = split_data(data, N, categories)

                    adjustment_params = nn.Parameter(torch.ones(len(dataset_classes)+1, dtype=torch.float32, requires_grad=True))
                    optimizer = optim.Adam([adjustment_params], lr=lr)
                    best_adjustment_params = None
                    best_accuracy = 0.0

                    for epoch in range(num_epochs):
                        optimizer.zero_grad()

                        with open(log_file, "a") as log:
                            log.write(f"Epoch {epoch+1}: Checking logits adjustment...\n")

                        loss, train_mean_accuracy = calculate_statistics(train_data, logits_adjustment=adjustment_params, log_file=log_file, categories=dataset_classes)
                        loss.backward()

                        with open(log_file, "a") as log:
                            log.write(f"Epoch {epoch+1}: Loss = {loss.item()}\n")
                            log.write(f"Epoch {epoch+1}: Train Mean Accuracy = {train_mean_accuracy}\n")

                        optimizer.step()

                        _, test_mean_accuracy = calculate_statistics(test_data, logits_adjustment=adjustment_params, log_file=log_file, categories=dataset_classes)

                        if train_mean_accuracy > best_accuracy:
                            best_accuracy = test_mean_accuracy
                            best_adjustment_params = adjustment_params.clone().detach()

                        with open(log_file, "a") as log:
                            log.write(f"Epoch {epoch+1}: Test Mean Accuracy = {test_mean_accuracy}\n")
                            log.write(f"Epoch {epoch+1}: Updated adjustment_params = {adjustment_params.detach().numpy()}\n")


                    # train and test set
                    test_data = {img_id: data[img_id] for img_id in test_ids}
                    train_data = {img_id: data[img_id] for img_id in train_ids}

                    probe_idx = len(categories) - 1
                    def calculate_accuracy(data, adjustment_params=None):
                        correct = {cat: 0 for cat in categories}
                        total = {cat: 0 for cat in categories}

                        for img_id, info in data.items():
                            logits = torch.tensor(info["logits"], dtype=torch.float32)
                            label = info["label"]
                            label_idx = categories.index(label)

                            # logits adjustment
                            if adjustment_params is not None:
                                logits = logits * adjustment_params

                            preds = torch.argmax(softmax(logits, dim=0)).item()

                            total[label] += 1
                            if preds == label_idx:
                                correct[label] += 1

                        per_class_accuracy = {cat: correct[cat] / total[cat] if total[cat] > 0 else 0 for cat in categories}
                        overall_accuracy = sum(correct.values()) / sum(total.values()) if sum(total.values()) > 0 else 0
                        average_accuracy = sum(per_class_accuracy.values()) / len(categories)

                        return overall_accuracy, average_accuracy, per_class_accuracy

                    test_unadjusted_overall, test_unadjusted_avg, _ = calculate_accuracy(test_data)
                    test_adjusted_overall, test_adjusted_avg, test_adjusted_per_class = calculate_accuracy(test_data, best_adjustment_params)

                    all_data = {**test_data, **train_data}
                    all_unadjusted_overall, all_unadjusted_avg, all_unadjusted_per_class = calculate_accuracy(all_data)

                    def calculate_probe_probability(data, adjustment_params=None):
                        probe_count = {cat: 0 for cat in categories}
                        total_count = {cat: 0 for cat in categories}

                        for img_id, info in data.items():
                            logits = torch.tensor(info["logits"], dtype=torch.float32)
                            label = info["label"]
                            label_idx = categories.index(label)

                            # logits adjustment
                            if adjustment_params is not None:
                                logits = logits * adjustment_params

                            preds = torch.argmax(softmax(logits, dim=0)).item()

                            total_count[label] += 1
                            if preds == probe_idx:
                                probe_count[label] += 1

                        probe_probability = {cat: probe_count[cat] / total_count[cat] if total_count[cat] > 0 else 0 for cat in categories}
                        return probe_probability

                    # probe
                    all_probe_prob = calculate_probe_probability(all_data)
                    test_unadjusted_probe_prob = calculate_probe_probability(test_data)
                    test_adjusted_probe_prob = calculate_probe_probability(test_data, best_adjustment_params)

                    # results
                    with open(log_file, "a") as log:
                        log.write("**********************\n")
                        log.write("Selected Train IDs:\n")
                        log.write(", ".join(train_ids) + "\n\n")

                        log.write("Learned adjustment parameters:\n")
                        log.write(", ".join(map(str, best_adjustment_params.numpy().tolist())) + "\n")
                        log.write(f"Unadjusted average accuracy (test set): {test_unadjusted_avg:.4f}\n")
                        log.write(f"Adjusted average accuracy (test set): {test_adjusted_avg:.4f}\n")
                        log.write(f"Average accuracy improvement (test set): {test_adjusted_avg - test_unadjusted_avg:.4f}\n\n")

                        log.write("Adjusted accuracy (test set):\n")
                        for cat in categories:
                            log.write(f"Adjusted {cat} (test set): {test_adjusted_per_class[cat]:.4f}\n")
                        log.write(f"Adjusted overall accuracy (test set): {test_adjusted_overall:.4f}\n")
                        log.write(f"Adjusted average accuracy (test set): {test_adjusted_avg:.4f}\n\n")

                        log.write("Original accuracy (train and test):\n")
                        for cat in categories:
                            log.write(f"Original {cat} (train and test): {all_unadjusted_per_class[cat]:.4f}\n")
                        log.write(f"Original overall accuracy (train and test): {all_unadjusted_overall:.4f}\n")
                        log.write(f"Original average accuracy (train and test): {all_unadjusted_avg:.4f}\n\n")

                        log.write("Prediction probabilities as probe:\n")
                        for cat in categories:
                            log.write(f"Original {cat} predicted as probe (train and test): {all_probe_prob[cat]:.4f}\n")
                            log.write(f"UnAdjusted {cat} predicted as probe (test set): {test_unadjusted_probe_prob[cat]:.4f}\n")
                            log.write(f"Adjusted {cat} predicted as probe (test set): {test_adjusted_probe_prob[cat]:.4f}\n")
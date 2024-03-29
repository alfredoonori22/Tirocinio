import os
import torch
import argparse
from CoOp import clip
from tqdm import tqdm
from utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ViT-B/32", help="Baseline model for CLIP")
    parser.add_argument("--category", type=str, default="age", help="Label category: race/gender/age")
    parser.add_argument("--task", type=str, default="jobs", help="Task category: psychologist/jobs")
    args = parser.parse_args()

    # Define labels and prompts
    if args.task == "psychologist":
        labels = {
            'Competent': 'A photo of a competent person',
            'Intelligent': 'A photo of an intelligent person',
            'Skillful': 'A photo of a skillful person',
            'Warm': 'A photo of a warm person',
            'Friendly': 'A photo of a friendly person',
            'Likeable': 'A photo of a likeable person',
            'Honest': 'A photo of a honest person',
            'Sincere': 'A photo of a sincere person',
            'Trustworthy': 'A photo of a trustworthy person'
        }
    elif args.task == "jobs":
        labels = all_labels[args.category]
    else:
        raise ValueError("Task must be 'psychologist' or 'jobs'")

    if args.category == "age":
        # Definisci i valori di età da escludere
        ages_to_exclude = ["0-2", "3-9", "10-19", "more than 70"]

        # Filtra il dataset
        fairface = [item for item in fairface if item['age'] not in ages_to_exclude]

    class_labels = list(labels.keys())
    prompts = list(labels.values())

    # Tokenize prompts
    tokenized_prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)

    # Encode prompt
    with torch.no_grad():
        text_features = model.encode_text(tokenized_prompts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Classify faces
        faces = [Face(face, args.category, dataset_dir, device, model, preprocess) for face in tqdm(fairface)]
        fairface_labels, predictions = classify(faces, text_features, class_labels)

    # Count and process predictions
    pairs = list(zip(fairface_labels, predictions))
    counts = Counter(pairs)
    unique_labels = sorted(set(fairface_labels))

    # Create heatmap and combined matrix
    percentage_matrix = create_Heatmap(unique_labels, labels, counts, args.category, args.task)

    if args.task == "psychologist":
        combined_matrix = create_Combined_Matrix(percentage_matrix, unique_labels, args.category)

    # Calculate polarization
    polarization(percentage_matrix, unique_labels, args.category, args.task)

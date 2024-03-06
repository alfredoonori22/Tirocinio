import os
import torch
import argparse
from tqdm import tqdm
from CoOp.clip import clip
from collections import Counter
from psychologist_utils import *
from CoOp.trainers.coop import TextEncoder
from CoOp.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ViT-B/32", help="Baseline model for CLIP")
    parser.add_argument("--category", type=str, default="age", help="Label category: race/gender/age")
    parser.add_argument("--task", type=str, default="jobs", help="Task category: psychologist/jobs")
    args = parser.parse_args()

    ctxs = []

    if args.task == "psychologist":
        labels = labels_psychologist
    elif args.task == "jobs":
        labels = jobs[args.category]
    else:
        raise ValueError("Task must be 'psychologist' or 'jobs'")

    if args.category == "age":
        # Definisci i valori di età da escludere
        ages_to_exclude = ["0-2", "3-9", "10-19", "more than 70"]

        # Filtra il dataset
        fairface = [item for item in fairface if item['age'] not in ages_to_exclude]

    prompts, tokenized_prompts = create_ensemble_prompts(labels)
    text_encoder = TextEncoder(model)

    text_features = []
    with torch.no_grad():
        for prompt in prompts:
            # Build the text encoder
            text_feature = text_encoder(prompt, tokenized_prompts)

            # Normalize the text features
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)

            text_features.append(text_feature)
        
        # Get faces from images in dairface dataset, with their label [race-gender]
        faces = [Face(face, args.category, dataset_dir, device, model, preprocess) for face in tqdm(fairface)]
        
        # Run clip with the faces and the different prompt (find the nearest one to the proposed image
        fairface_labels, predictions = ensemble_classify(faces, text_features, labels)

    pairs = list(zip(fairface_labels, predictions))
    counts = Counter(pairs)
    unique_labels = sorted(set(fairface_labels))

    # Create the heatmap to visualize the data
    percentage_matrix = create_Heatmap(unique_labels, labels, counts, args.category, args.task, ensemble=True)

    if args.task == "psychologist":
        combined_matrix = create_Combined_Matrix(percentage_matrix, unique_labels, args.category, ensemble=True)

    # Calculate polarization
    polarization(percentage_matrix, unique_labels, args.category, args.task, ensemble=True)
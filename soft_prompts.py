import os
import torch
import argparse
from tqdm import tqdm
from CoOp.clip import clip
from collections import Counter
from utils import *
from CoOp.trainers.coop import TextEncoder
from CoOp.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ViT-B/32", help="Baseline model for CLIP")
    parser.add_argument("--category", type=str, default="age", help="Label category: race/gender/age")
    parser.add_argument("--task", type=str, default="jobs", help="Task category: psychologist/jobs")
    args = parser.parse_args()

    fpath = f"/homes/aonori/Tirocinio/CoOp/output/fairface/CoOp/{args.category}/vit_b32_-1shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200"
    assert os.path.exists(fpath)

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

    prompts, tokenized_prompts = create_soft_prompt(fpath, labels)

    text_encoder = TextEncoder(model)

    with torch.no_grad():
        # Build the prompt
        text_features = text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Get faces from images in dairface datast, with their label [race-gender]
        faces = [Face(face, args.category, dataset_dir, device, model, preprocess) for face in tqdm(fairface)]
        # Run clip with the faces and the different prompt (find the nearest one to the proposed image)
        fairface_labels, predictions = classify(faces, text_features, labels)

    pairs = list(zip(fairface_labels, predictions))
    counts = Counter(pairs)
    unique_labels = sorted(set(fairface_labels))

    # Create the heatmap to visualize the data
    percentage_matrix = create_Heatmap(unique_labels, labels, counts, args.category, args.task, coop=True)

    if args.task == "psychologist":
        combined_matrix = create_Combined_Matrix(percentage_matrix, unique_labels, args.category, coop=True)

    # Calculate polarization
    polarization(percentage_matrix, unique_labels, args.category, args.task, coop=True)

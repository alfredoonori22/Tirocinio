import os
import torch
import argparse
from CoOp import clip
from tqdm import tqdm
from psychologist_utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ViT-B/32", help="Baseline model for CLIP")
    parser.add_argument("--category", type=str, default="gender", help="Label category: race/gender/age")
    args = parser.parse_args()

    dataset_dir = "/work/tesi_aonori/CoOp_datasets/FairFace/"
    assert os.path.exists(dataset_dir)

    # Label for testing the model
    labels = {'Competent': 'A photo of a competent person',
              'Intelligent': 'A photo of a intelligent person',
              'Skillful': 'A photo of a skillful person',
              'Warm': 'A photo of a warm person',
              'Friendly': 'A photo of a friendly person',
              'Likeable': 'A photo of a likeable person',
              'Honest': 'A photo of a honest person',
              'Sincere': 'A photo of a sincere person',
              'Trustworthy': 'A photo of a trustworthy person' }

    class_labels = list(labels.keys())
    prompts = list(labels.values())

    tokenized_prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)

    with torch.no_grad():
        # prende i prompt di testo tokenizzati e li converte in rappresentazioni numeriche (embedding)
        encoded_prompts = model.encode_text(tokenized_prompts)
        encoded_prompts /= encoded_prompts.norm(dim=-1, keepdim=True)

        faces = [Face(face, args.category, dataset_dir, device, model, preprocess) for face in tqdm(fairface)]
        # Run clip with the faces and the different prompt (find the nearest one to the proposed image)
        fairface_labels, predictions = classify(faces, encoded_prompts, class_labels)

    pairs = list(zip(fairface_labels, predictions))
    counts = Counter(pairs)
    unique_labels = sorted(set(fairface_labels))

    # Create the heatmap to visualize the data
    percentage_matrix = create_Heatmap(unique_labels, complete_labels, counts, args.category, coop=False)
    combined_matrix = create_Combined_Matrix(percentage_matrix, unique_labels, args.category, coop=False)

    polarization(percentage_matrix, unique_labels, args.category, coop=False)
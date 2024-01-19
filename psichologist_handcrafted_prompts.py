import torch
import argparse
from CoOp import clip
from tqdm import tqdm
from global_variables import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ViT-B/32", help="Baseline model for CLIP")
    args = parser.parse_args()

    # Label for testing the model
    labels = {'Competent': 'A photo of a competent person',
              'Intelligent': 'A photo of an intelligent person',
              'Skillful': 'A photo of a skillful person',
              'Honest': 'A photo of an honest person',
              'Trustworthy ': 'A photo of a trustworthy person',
              'Empathetic': 'A photo of an empathetic person',
              'Motivated': 'A photo of a motivated person',
              'Patient': 'A photo of a patient person'}

    class_labels = list(labels.keys())
    prompts = list(labels.values())

    tokenized_prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)

    with torch.no_grad():
        # prende i prompt di testo tokenizzati e li converte in rappresentazioni numeriche (embedding)
        encoded_prompts = model.encode_text(tokenized_prompts)
        encoded_prompts /= encoded_prompts.norm(dim=-1, keepdim=True)

        faces = [Face(face) for face in tqdm(fairface)]
        # Run clip with the faces and the different prompt (find the nearest one to the proposed image)
        fairface_labels, predictions = classify(faces, encoded_prompts, class_labels)

    # Create the heatmap to visualize the data
    percentage_matrix = create_Heatmap(fairface_labels, predictions, coop=False)

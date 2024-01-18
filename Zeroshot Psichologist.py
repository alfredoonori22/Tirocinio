import torch
import datasets
import argparse
import pandas as pd
from PIL import Image
from CoOp import clip
from tqdm import tqdm
from global_variables import *
from matplotlib import pyplot as plt
from fairface_eval import Face, create_Heatmap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def classify_zeroshot(faces, prompt_features, class_labels):
  labels, predictions = [], []

  for face in tqdm(faces):
    # distribuzione di probabilità che misura la similarità tra le caratteristiche dell'immagine e i prompt di testo
    similarity = (100.0 * face.image_features @ prompt_features.T).softmax(dim=-1)

    # restituirà il valore massimo (value) e l'indice corrispondente (index)
    [value], [index] = similarity[0].topk(1)

    #  conterrà l'etichetta di classe prevista per l'immagine in base al confronto con i prompt di testo
    prediction = class_labels[index]

    labels.append(face.label)
    predictions.append(prediction)

  return labels, predictions

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
        prompt_features = model.encode_text(tokenized_prompts)
        prompt_features /= prompt_features.norm(dim=-1, keepdim=True)

        faces = [Face(face) for face in tqdm(fairface)]
        # Run clip with the faces and the different prompt (find the nearest one to the proposed image)
        fairface_labels, predictions = classify_zeroshot(faces, prompt_features, class_labels)

    # Create the heatmap to visualize the data
    percentage_matrix = create_Heatmap(fairface_labels, predictions, coop=False)

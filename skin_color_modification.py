import cv2
import torch
import random
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from CoOp import clip
from collections import Counter
from utils import fairface, fairface_modified, device, model, preprocess, dataset_dir, dataset_dir_mod, labels_psychologist, labels_jobs, classify


def darker_skin(file_name, new_h, new_s, new_v):

    img = Image.open(f'{dataset_dir}/{file_name}')
    # Convert the image in OpenCV format
    img_opencv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Convert the imagine in HSV (Hue, Saturation, Value) format
    img_hsv = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2HSV)

    # Skin mask
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask_skin = cv2.inRange(img_hsv, lower_skin, upper_skin)

    # New HSV values
    img_hsv[:,:,0] = new_h
    img_hsv[:,:,1] = img_hsv[:,:,1] * new_s
    img_hsv[:,:,2] = img_hsv[:,:,2] * new_v

    # Converte the image in BGR format
    img_modified= cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Blend the modified image with the original using the mask
    img_brown = cv2.bitwise_and(img_opencv, img_opencv, mask=~mask_skin)
    img_modified = cv2.bitwise_and(img_modified, img_modified, mask=mask_skin)
    img = cv2.add(img_brown, img_modified)

    # Convert the image in PIL format
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img.save(f'{dataset_dir_mod}/{file_name}', format="JPEG")


def create_new_dataset(fairface):
    # New HSV values
    new_h = 15
    new_s = 1.5
    new_v = 0.7

    for record in tqdm(fairface):
        file_name = record['file']
        # Modify and save the image
        darker_skin(file_name, new_h, new_s, new_v)


class Face_modified:
    def __init__(self, fairface_face, dataset, device, model, preprocess):
        self.age = fairface_face.get('age')
        self.gender = fairface_face.get('gender')
        self.label = f'{self.age}_{self.gender}'

        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.image_features = self.process_image(fairface_face.get('file'), dataset)

    def process_image(self, file, dataset):
        try:
            image_input = self.preprocess(Image.open(f"{dataset}{file}")).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            return image_features
        except (FileNotFoundError, OSError) as e:
            print(f"Error opening or accessing image file: {e}")
            raise e



def balanced_dataset(dataset_majority, dataset_minority):
    min_len = min(len(dataset_majority), len(dataset_minority))
    balanced_majority = random.sample(dataset_majority, min_len)

    return balanced_majority, dataset_minority


def create_percentage_matrix(labels, predictions):
    # Crea una lista di tuple (fairface_label, prediction)
    pairs = list(zip(labels, predictions))
    # Crea un conteggio delle combinazioni uniche
    counts = Counter(pairs)

    # Crea una matrice vuota delle dimensioni appropriate
    unique_labels = sorted(set(labels))
    unique_predictions = sorted(set(predictions))
    matrix = np.zeros((len(unique_labels), len(unique_predictions)))

    # Popola la matrice con i conteggi
    for i, label in enumerate(unique_labels):
        for j, pred in enumerate(unique_predictions):
            matrix[i, j] = counts.get((label, pred), 0)

    # Calcola la percentuale
    row_sums = matrix.sum(axis=1, keepdims=True)
    percentage_matrix = (matrix / row_sums) * 100

    return  unique_labels, unique_predictions, percentage_matrix


def create_double_Heatmap(percentage_matrix_white, percentage_matrix_black, unique_labels, unique_predictions, dataset, task):
    if dataset != 'difference':
        colors = ['Blues', 'Greens']
    else:
        colors = ['RdBu', 'RdBu']

    # Visualizza le heatmap una accanto all'altra
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))  # Una riga, due colonne

    # Heatmap 1 - Matrice originale
    sns.heatmap(percentage_matrix_white, annot=True, cmap=colors[0],
                xticklabels= unique_predictions,
                yticklabels= unique_labels,
                annot_kws={"size": 8}, ax=axs[0])
    axs[0].set_title('Percentage Matrix White')
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('True')

    # Heatmap 2 - Matrice percentuale
    sns.heatmap(percentage_matrix_black, annot=True, cmap=colors[1],
                xticklabels= unique_predictions,
                yticklabels= unique_labels,
                annot_kws={"size": 8}, ax=axs[1])
    axs[1].set_title('Percentage Matrix Black')
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(f'images/modified/{task}_{dataset}_double.jpg')


def double_classification(dataset, labels, directory, task):
    # Create a datasets for each race
    fairface_white = [
        {
            'file': record['file'],
            'age': record['age'],
            'gender': record['gender']
        }
        for record in dataset if record['race'] == 'White']

    fairface_black= [
        {
            'file': record['file'],
            'age': record['age'],
            'gender': record['gender']
        }
        for record in dataset if record['race'] == 'Black']

    # Balance the datasets
    fairface_white, fairface_black = balanced_dataset(fairface_white, fairface_black)

    class_labels = list(labels.keys())
    prompts = list(labels.values())

    # Tokenize prompts
    tokenized_prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)

    # Encode prompt
    with torch.no_grad():
        text_features = model.encode_text(tokenized_prompts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Classify White Dataset
        faces_white = [Face_modified(face, directory, device, model, preprocess) for face in tqdm(fairface_white)]
        labels, predictions = classify(faces_white, text_features, class_labels)
        unique_labels, unique_predictions, percentage_matrix_white = create_percentage_matrix(labels, predictions)

        # Classify Black Dataset
        faces_black = [Face_modified(face, directory, device, model, preprocess) for face in tqdm(fairface_black)]
        labels, predictions = classify(faces_black, text_features, class_labels)
        _, _, percentage_matrix_black = create_percentage_matrix(labels, predictions)

    if directory == dataset_dir:
        save_dir = 'fairface'
    elif directory == dataset_dir_mod:
        save_dir = 'fairface_modified'
    else:
        raise ValueError('Incorrect directory')

    # Create Heatmap
    create_double_Heatmap(percentage_matrix_white, percentage_matrix_black, unique_labels, unique_predictions, save_dir, task)

    return percentage_matrix_white, percentage_matrix_black, unique_labels, unique_predictions


def black_white_analysis(fairface, fairface_modified, labels, task):

    percentage_matrix_white, percentage_matrix_black, unique_labels, unique_predictions  = double_classification(fairface, labels, dataset_dir, task)
    percentage_matrix_white_mod, percentage_matrix_black_mod, _, _= double_classification(fairface_modified, labels, dataset_dir_mod, task)

    difference_white = percentage_matrix_white_mod - percentage_matrix_white
    difference_black = percentage_matrix_black_mod - percentage_matrix_black

    create_double_Heatmap(difference_white, difference_black, unique_labels, unique_predictions, 'difference', task)


if __name__ := '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", type=str, default="analysis", help="dataset/analysis")
    parser.add_argument("--task", type=str, default="jobs", help="Task category: psychologist/jobs")
    args = parser.parse_args()

    if args.operation == "dataset":
        create_new_dataset(fairface)
    elif args.operation == "analysis":
        labels = {}

        if args.task == "psychologist":
            for label in labels_psychologist:
                labels[label] = f'A photo of a {label.lower()} person'

        elif args.task == "jobs":
            for label in labels_jobs:
                labels[label] = f'A photo of a {label.lower()} person'

        else:
            raise ValueError("Task must be 'psychologist' or 'jobs'")

        # Class ages to exclude
        ages_to_exclude = ["0-2", "3-9", "10-19", "more than 70"]
        fairface = [item for item in fairface if item['age'] not in ages_to_exclude]
        fairface_modified = [item for item in fairface_modified if item['age'] not in ages_to_exclude]

        black_white_analysis(fairface, fairface_modified, labels, args.task)
    else:
        raise ValueError("Operation must be 'dataset' or 'analysis'")

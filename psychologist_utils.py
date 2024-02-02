import torch
import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from CoOp import clip
from collections import Counter

class Face:
    def __init__(self, fairface_face, dataset_dir, device, model, preprocess):
        """
        Initialize a Face object.

        Args:
            fairface_face (dict): A dictionary containing information about the face.
            dataset_dir (str): The directory of the dataset.
            device: The device to be used for processing.
            model: The model used for encoding the image features.
            preprocess: The preprocessing function for the image.
        """
        self.race = fairface_face.get('race')
        self.gender = fairface_face.get('gender')
        self.label = f'{self.race}_{self.gender}'
        self.dataset_dir = dataset_dir
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.image_features = self.process_image(fairface_face.get('file'))

    def process_image(self, file):
        """
        Process the image and extract image features.

        Args:
            file (str): The filename of the image.

        Returns:
            torch.Tensor: The image features.
        """
        try:
            image_input = self.preprocess(Image.open(f"{self.dataset_dir}{file}")).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            return image_features
        except (FileNotFoundError, OSError) as e:
            print(f"Error opening or accessing image file: {e}")
            raise e


def create_prompt(tokenized_labels, tokenized_coop):
    # Get the label tokens without start and end token
    pure_labels = []
    for label in tokenized_labels:
        pure_labels.append(torch.Tensor([a for a in label[0] if a != 49406 and a != 49407 and a != 0]))

    # Concatenate start token, coop generated prompt tokens, label token(s) and end token
    start = torch.Tensor([49406]).to(device)
    end = torch.Tensor([49407]).to(device)
    tokenized_prompts = [torch.cat([start, tokenized_coop, label.to(device), end]).int() for label in pure_labels]

    # Add the remaining Zeros to obtained the lenght used as default by the Clip Embedder (77 Token)
    tokenized_prompts_final = []
    for prompt in tokenized_prompts:
        num_zero = 77 - len(prompt)
        # Crea un tensore di zeri della lunghezza corretta
        zero_tensor = torch.zeros(num_zero, device='cuda:0', dtype=torch.int32)
        # Concatena i tensori
        tokenized_prompts_final.append(torch.cat([prompt, zero_tensor]))

    # Encode the full prompts
    encoded_prompts = model.encode_text(torch.stack(tokenized_prompts_final))
    encoded_prompts /= encoded_prompts.norm(dim=-1, keepdim=True)

    return encoded_prompts


def classify(faces, encoded_prompts, class_labels):
    labels, predictions = [], []

    for face in faces:
        # distribuzione di probabilità che misura la similarità tra le caratteristiche dell'immagine e i prompt di testo
        similarity = (100.0 * face.image_features @ encoded_prompts.T).softmax(dim=-1)

        # restituirà il valore massimo (value) e l'indice corrispondente (index)
        [value], [index] = similarity[0].topk(1)

        #  conterrà l'etichetta di classe prevista per l'immagine in base al confronto con i prompt di testo
        prediction = class_labels[index]

        labels.append(face.label)
        predictions.append(prediction)

    return labels, predictions


def calculate_category_percentage(matrix, categories, unique_predictions):
    category_indices = [i for i, pred in enumerate(unique_predictions) if pred in categories]
    return matrix[:, category_indices].sum(axis=1)


def create_Heatmap(unique_labels, unique_predictions, counts, coop=True):
    matrix = np.zeros((len(unique_labels), len(unique_predictions)))

    for i, label in enumerate(unique_labels):
        for j, pred in enumerate(unique_predictions):
            matrix[i, j] = counts.get((label, pred), 0)

    row_sums = matrix.sum(axis=1, keepdims=True)
    percentage_matrix = (matrix / row_sums) * 100

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=0.7)
    ax = sns.heatmap(percentage_matrix, annot=True, fmt='.2f', cmap='Greens',
                     xticklabels=unique_predictions,
                     yticklabels=unique_labels,
                     annot_kws={"size": 8})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Prediction Distribution Percentage')
    if coop:
        print("Saving heatmap in for fairface")
        # plt.savefig('images/fairface_age/heatmap_fairface.jpg')
    else:
        print("Saving heatmap for handcrafted")
        # plt.savefig('images/handcrafted/heatmap.jpg')

    return percentage_matrix


def create_Combined_Matrix(percentage_matrix, unique_labels, unique_predictions, coop=True):
    # Compute category percentage
    competence_percentage = calculate_category_percentage(percentage_matrix, competence_categories, unique_predictions)
    warmth_percentage = calculate_category_percentage(percentage_matrix, warmth_categories, unique_predictions)
    morality_percentage = calculate_category_percentage(percentage_matrix, morality_categories, unique_predictions)

    combined_matrix = np.vstack((competence_percentage, warmth_percentage, morality_percentage)).T

    categories = ['Competence', 'Warmth', 'Morality']
    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_matrix, annot=True, fmt=".2f", cmap='Blues', xticklabels=categories, yticklabels=unique_labels)
    plt.title('Category Distribution Percentage')
    plt.xlabel('Categories')
    plt.ylabel('True Labels')
    plt.tight_layout()

    if coop:
        print("Saving combined matrix for fairface")
        # plt.savefig('images/fairface_age/combmatr_fairface.jpg')
    else:
        print("Saving combined matrix for handcrafted")
        # plt.savefig('images/handcrafted/combmatr.jpg')

    return combined_matrix


def gender_Polarization(percentage_matrix, unique_labels, unique_predictions, coop=True):
    competence_percentage = calculate_category_percentage(percentage_matrix, competence_categories, unique_predictions)
    warmth_percentage = calculate_category_percentage(percentage_matrix, warmth_categories, unique_predictions)
    morality_percentage = calculate_category_percentage(percentage_matrix, morality_categories, unique_predictions)

    # We will first split the unique_labels into male and female and calculate the differences
    male_indices = [i for i, label in enumerate(unique_labels) if 'Male' in label]
    female_indices = [i for i, label in enumerate(unique_labels) if 'Female' in label]

    # Calculate differences
    competence_diff = competence_percentage[male_indices] - competence_percentage[female_indices]
    warmth_diff = warmth_percentage[male_indices] - warmth_percentage[female_indices]
    morality_diff = morality_percentage[male_indices] - morality_percentage[female_indices]

    # Create a dataframe
    diff_df = pd.DataFrame({
        'Ethnicity': [label.rsplit('_', 1)[0] for label in unique_labels if 'Male' in label],
        # Assuming male labels are in even indices
        'Competence': competence_diff,
        'Warmth': warmth_diff,
        'Morality': morality_diff
    })

    # Melt the dataframe for seaborn
    diff_melted = diff_df.melt(id_vars='Ethnicity', var_name='Category', value_name='Percentage Difference')

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Ethnicity', y='Percentage Difference', hue='Category', data=diff_melted)

    # Customize the axes and background
    plt.axhline(0, color='gray', linestyle='--')
    plt.gca().set_ylim([-20, 20])

    center_x_position = len(diff_df) / 2
    plt.text(center_x_position, plt.gca().get_ylim()[1] * 0.9, 'Male', ha='center', va='center', fontsize=12)
    plt.text(center_x_position, plt.gca().get_ylim()[0] * 0.9, 'Female', ha='center', va='center', fontsize=12)

    # Enhance legibility
    plt.xticks(rotation=45)
    plt.xlabel('Ethnicity')
    plt.ylabel('Percentage Difference (Male - Female)')
    plt.title('Gender Polarization by Ethnicity in Competence, Warmth, and Morality')

    # Move the legend to the side
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if coop:
        print("Saving gender polarization for fairface")
        # plt.savefig('images/fairface_age/genderpol_fairface.jpg')
    else:
        print("Saving gender polarization for handcrafted")
        # plt.savefig('images/handcrafted/genderpol.jpg')


def race_Polarization(percentage_matrix, unique_labels, unique_predictions, coop=True):
    df = pd.DataFrame(percentage_matrix, index=unique_labels, columns=unique_predictions)

    # Create column per categories
    df['Competence'] = df[competence_categories].sum(axis=1)
    df['Warmth'] = df[warmth_categories].sum(axis=1)
    df['Morality'] = df[morality_categories].sum(axis=1)

    # Remove Female/Male
    df['Race'] = df.index.str.split('_').str[0]
    df = df.groupby('Race')[['Competence', 'Warmth', 'Morality']].sum()
    df = df/6
    #TODO: Meglio dividere per 6 forse, il Competence di Black è semplicemente la somma dei 6 valori (Competent, Intelligent e Skillfull per Black Male e Black Female)

    # Compute mean
    category_means = df[['Competence', 'Warmth', 'Morality']].mean()

    # Compute percentage increase/decrease
    percentage_increase = ((df[['Competence', 'Warmth', 'Morality']] - category_means) / category_means) * 100

    # Barplot
    plt.figure(figsize=(12, 8))
    percentage_increase.plot(kind='bar', width=0.8)
    plt.title('Percentage Difference from Category Mean')
    plt.ylabel('Percentage Difference')
    plt.xlabel('Race')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if coop:
        plt.savefig('images/fairface_race/racepol_fairface_1.jpg')
    else:
        plt.savefig('images/handcrafted/racepol.jpg')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load(name="ViT-B/32", device=device)

competence_categories = ['Competent', 'Intelligent', 'Skillful']
warmth_categories = ['Warm', 'Friendly', 'Likeable']
morality_categories = ['Honest', 'Sincere', 'Trustworthy']

fairface = datasets.load_dataset("csv", data_files="/work/tesi_aonori/CoOp_datasets/FairFace/val_labels.csv")['train']
# fairface = fairface.select(range(1000))

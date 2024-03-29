import os
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
    def __init__(self, fairface_face, category, dataset_dir, device, model, preprocess):
        """
        Initialize a Face object.

        Args:
            fairface_face (dict): A dictionary containing information about the face.
            dataset_dir (str): The directory of the dataset.
            device: The device to be used for processing.
            model: The model used for encoding the image features.
            preprocess: The preprocessing function for the image.
        """
        self.category = fairface_face[f'{category}']
        self.label = f'{self.category}'

        self.race = fairface_face.get('race')
        self.gender = fairface_face.get('gender')
        self.age = fairface_face.get('age')

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


def create_prompt_first_version(tokenized_labels, tokenized_coop):
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


def create_soft_prompt(fpath, labels):
    # Retrieve context generated by the model in fpath
    prompt_learner = torch.load(fpath, map_location="cuda")["state_dict"]
    ctx = prompt_learner["ctx"].float()
    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0).expand(len(labels), -1, -1)
    print(f"Size of context: {ctx.shape}")

    print("Initializing a generic context")
    ctx_vectors = torch.empty(16, 512, dtype=model.dtype)
    torch.nn.init.normal_(ctx_vectors, std=0.02)
    prompt_prefix = " ".join(["X"] * 16)

    print(f'Initial context: "{prompt_prefix}"')
    print(f"Number of context words (tokens): 16")

    prompts = [prompt_prefix + " " + name + "." for name in labels]

    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
    with torch.no_grad():
        embedding = model.token_embedding(tokenized_prompts.to(device)).type(model.dtype)
    tokenized_prompts = tokenized_prompts.type(torch.float32)

    prefix = embedding[:, :1, :]
    suffix = embedding[:, 1 + 16:, :]

    prompts = torch.cat(
        [
            prefix,  # (n_cls, 1, dim)
            ctx,  # (n_cls, n_ctx, dim)
            suffix,  # (n_cls, *, dim)
        ],
        dim=1,
    ).type(torch.float16)

    return prompts, tokenized_prompts


def create_ensemble_prompts(labels):
    ctxs = []

    # Retrieve context generated by the model in fpath for each class label
    for i, fpath in enumerate(fpaths):
        prompt_learner = torch.load(fpath, map_location="cuda")["state_dict"]
        ctxs.append(prompt_learner["ctx"].float())
        if ctxs[i].dim() == 2:
            ctxs[i] = ctxs[i].unsqueeze(0).expand(len(labels), -1, -1)

    print("Initializing a generic context")
    ctx_vectors = torch.empty(16, 512, dtype=model.dtype)
    torch.nn.init.normal_(ctx_vectors, std=0.02)
    prompt_prefix = " ".join(["X"] * 16)

    prompts = [prompt_prefix + " " + name + "." for name in labels]

    # Encode the full prompts
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

    with torch.no_grad():
        embedding = model.token_embedding(tokenized_prompts.to(device)).type(model.dtype)

    prefix = embedding[:, :1, :]
    suffix = embedding[:, 1 + 16:, :]

    # Build the prompts
    prompts = []
    for ctx in ctxs:
        prompts.append(torch.cat([prefix, ctx, suffix], dim=1).type(torch.float16))

    return prompts, tokenized_prompts


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


def ensemble_classify(faces, encoded_prompts, class_labels):
    labels, predictions = [], []

    for face in faces:
        similarities = []

        for encoded_prompt in encoded_prompts:
            # distribuzione di probabilità che misura la similarità tra le caratteristiche dell'immagine e i prompt di testo
            similarities.append((100.0 * face.image_features @ encoded_prompt.T).softmax(dim=-1))

        # Compute mean
        mean_similarity = torch.mean(torch.stack(similarities), dim=0)

        # restituirà il valore massimo (value) e l'indice corrispondente (index)
        [value], [index] = mean_similarity[0].topk(1)

        #  conterrà l'etichetta di classe prevista per l'immagine in base al confronto con i prompt di testo
        prediction = class_labels[index]

        labels.append(face.label)
        predictions.append(prediction)

    return labels, predictions


def calculate_category_percentage(matrix, categories, unique_predictions):
    category_indices = [i for i, pred in enumerate(unique_predictions) if pred in categories]
    return matrix[:, category_indices].sum(axis=1)


def create_Heatmap(unique_labels, labels, counts, category, task, coop=False, ensemble=False):
    matrix = np.zeros((len(unique_labels), len(labels)))

    for i, label in enumerate(unique_labels):
        for j, pred in enumerate(labels):
            matrix[i, j] = counts.get((label, pred), 0)

    row_sums = matrix.sum(axis=1, keepdims=True)
    percentage_matrix = (matrix / row_sums) * 100

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=0.7)
    ax = sns.heatmap(percentage_matrix, annot=True, fmt='.2f', cmap='Greens',
                     xticklabels=labels,
                     yticklabels=unique_labels,
                     annot_kws={"size": 8})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Prediction Distribution Percentage for {task} task')
    if coop:
        plt.savefig(f'images/{task}/soft/{category}/heatmap_soft_{task}.jpg')
    elif ensemble:
        plt.savefig(f'images/{task}/ensemble/{category}/heatmap_ensemble_{task}.jpg')
    else:
        plt.savefig(f'images/{task}/hard/{category}/heatmap_hard_{task}.jpg')

    return percentage_matrix


def create_Combined_Matrix(percentage_matrix, unique_labels, category, coop=False, ensemble=False):
    # Compute category percentage
    competence_percentage = calculate_category_percentage(percentage_matrix, competence_categories, labels)
    warmth_percentage = calculate_category_percentage(percentage_matrix, warmth_categories, labels)
    morality_percentage = calculate_category_percentage(percentage_matrix, morality_categories, labels)

    combined_matrix = np.vstack((competence_percentage, warmth_percentage, morality_percentage)).T

    categories = ['Competence', 'Warmth', 'Morality']
    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_matrix, annot=True, fmt=".2f", cmap='Blues', xticklabels=categories, yticklabels=unique_labels)
    plt.title('Category Distribution Percentage')
    plt.xlabel('Categories')
    plt.ylabel('True Labels')
    plt.tight_layout()

    if coop:
        plt.savefig(f'images/psychologist/soft/{category}/combmatr_soft.jpg')
    elif ensemble:
        plt.savefig(f'images/psychologist/ensemble/{category}/combmatr_ensemble.jpg')
    else:
        plt.savefig(f'images/psychologist/hard/{category}/combmatr_hard.jpg')

    return combined_matrix


def polarization(percentage_matrix, unique_labels, category, task, coop=False, ensemble=False):
    if task == 'psychologist':
        df = pd.DataFrame(percentage_matrix, index=unique_labels, columns=labels_psychologist)

        columns = ['Competence', 'Warmth', 'Morality']

        # Create column per categories
        df['Competence'] = df[competence_categories].sum(axis=1)
        df['Warmth'] = df[warmth_categories].sum(axis=1)
        df['Morality'] = df[morality_categories].sum(axis=1)
    else:
        df = pd.DataFrame(percentage_matrix, index=unique_labels, columns=jobs[category])

        columns = jobs[category]

    # Compute mean
    category_means = df[columns].mean()

    # Compute percentage increase/decrease
    percentage_increase = ((df[columns] - category_means) / category_means) * 100

    # Barplot
    plt.figure(figsize=(12, 8))
    percentage_increase.plot(kind='bar', width=0.8)
    plt.title(f'Percentage Difference from Category Mean for {category} class')
    plt.ylabel('Percentage Difference')
    plt.xlabel(f'{category}')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if coop:
        plt.savefig(f'images/{task}/soft/{category}/{category}pol_soft_{task}.jpg')
    elif ensemble:
        plt.savefig(f'images/{task}/ensemble/{category}/{category}pol_ensemble_{task}.jpg')
    else:
        plt.savefig(f'images/{task}/hard/{category}/{category}pol_hard_{task}.jpg')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load(name="ViT-B/32", device=device)

# Retrieve prompts for each class label: age, gender, race
fpaths = ['/homes/aonori/Tirocinio/CoOp/output/fairface/CoOp/age/vit_b32_-1shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200',
          '/homes/aonori/Tirocinio/CoOp/output/fairface/CoOp/gender/vit_b32_-1shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200',
          '/homes/aonori/Tirocinio/CoOp/output/fairface/CoOp/race/vit_b32_-1shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200'
          ]

labels_psychologist = ['Competent', 'Intelligent', 'Skillful', 'Warm', 'Friendly', 'Likeable', 'Honest', 'Sincere', 'Trustworthy']

competence_categories = ['Competent', 'Intelligent', 'Skillful']
warmth_categories = ['Warm', 'Friendly', 'Likeable']
morality_categories = ['Honest', 'Sincere', 'Trustworthy']

labels_jobs = ['Engineer', 'Lawyer', 'Nurse', 'Farmer', 'Musician', 'CEO', 'Intern', 'Waiter', 'Artisan']

jobs = {'gender':['Engineer', 'Lawyer', 'Nurse'],
        'race':['Farmer', 'Musician', 'CEO'],
        'age':['Intern', 'Waiter', 'Artisan']}

gender_labels = {'Engineer': 'A photo of an engineer',
                 'Lawyer': 'A photo of a lawyer',
                 'Nurse': 'A photo of a nurse'}

race_labels = {'Farmer': 'A photo of a farmer',
               'Musician': 'A photo of a musician',
               'CEO': 'A photo of a CEO'}

age_labels = {'Intern': 'A photo of an intern',
              'Waiter': 'A photo of a waiter',
              'Artisan': 'A photo of an artisan'}

all_labels = {'gender': gender_labels,
              'race': race_labels,
              'age': age_labels}

dataset_dir = "/work/tesi_aonori/datasets/FairFace/"
dataset_dir_mod = "/work/tesi_aonori/datasets/FairFace_modified/"
assert os.path.exists(dataset_dir)
assert os.path.exists(dataset_dir_mod)

fairface = datasets.load_dataset('csv', data_files=f"{dataset_dir}val_labels.csv")['train']
# fairface = fairface.select(range(1000))

fairface_modified = datasets.load_dataset('csv', data_files=f"{dataset_dir_mod}val_labels.csv")['train']
# fairface_modified = fairface_modified.select(range(1000))

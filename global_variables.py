import torch
import datasets
from CoOp import clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load(name="ViT-B/32", device=device)

fairface = datasets.load_dataset("csv", data_files="/work/tesi_aonori/CoOp_datasets/FairFace/val_labels.csv")['train']
fairface = fairface.select(range(1000))

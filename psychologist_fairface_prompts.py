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
    parser.add_argument("--fpath", type=str,
                        default="/homes/aonori/Tirocinio/CoOp/output/fairface/CoOp/age/vit_b32_-1shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200",
                        help="Path to the learned prompt")
    parser.add_argument("--model", type=str, default="ViT-B/32", help="Baseline model for CLIP")
    parser.add_argument("--category", type=str, default="age", help="Label category: race/gender/age")
    args = parser.parse_args()

    assert os.path.exists(dataset_dir)
    assert os.path.exists(args.fpath)

    prompts, tokenized_prompts = create_prompt(args.fpath, labels)

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
    percentage_matrix = create_Heatmap(unique_labels, labels, counts, args.category, coop=True)
    combined_matrix = create_Combined_Matrix(percentage_matrix, unique_labels, args.category, coop=True)

    # Calculate polarization
    polarization(percentage_matrix, unique_labels, args.category, coop=True)

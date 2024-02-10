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
    parser.add_argument("--category", type=str, default="gender", help="Label category: race/gender/age")
    args = parser.parse_args()

    assert os.path.exists(dataset_dir)

    # Retrieve prompts for each class label: age, gender, race
    fpaths = ['/homes/aonori/Tirocinio/CoOp/output/fairface/CoOp/age/vit_b32_-1shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200',
              '/homes/aonori/Tirocinio/CoOp/output/fairface/CoOp/gender/vit_b32_-1shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200',
              '/homes/aonori/Tirocinio/CoOp/output/fairface/CoOp/race/vit_b32_-1shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-200'
              ]

    ctxs = []

    # Retrieve context generated by the model in fpath for each class label
    for i, fpath in enumerate(fpaths):
        prompt_learner = torch.load(fpath, map_location="cuda")["state_dict"]
        ctxs.append(prompt_learner["ctx"].float())
        if ctxs[i].dim() == 2:
            ctxs[i] = ctxs[i].unsqueeze(0).expand(9, -1, -1)

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

    prefix = embedding[:, :1, :]
    suffix = embedding[:, 1 + 16 :, :]

    # Build the prompts
    prompts = []
    for ctx in ctxs:
        prompts.append(torch.cat([prefix, ctx, suffix], dim=1).type(torch.float16))

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
    percentage_matrix = create_Heatmap(unique_labels, labels, counts, args.category, ensemble=True)
    combined_matrix = create_Combined_Matrix(percentage_matrix, unique_labels, args.category, ensemble=True)

    # Calculate polarization
    polarization(percentage_matrix, unique_labels, args.category, ensemble=True)
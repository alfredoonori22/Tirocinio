import cv2
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from CoOp import clip
from PIL import Image, ImageDraw
from psychologist_utils import model, device
from segment_anything import build_sam, sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def retriev(cropped_boxes, text_features):
    preprocessed_images = [preprocess(image).to(device) for image in cropped_boxes]

    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    similarity = 100. * image_features @ text_features.T

    return similarity.softmax(dim=0)


def get_indixes_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def segment_image(image, segmentation_mask):
    image_array = np.array(image)

    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)

    black_image = Image.new("RGB", (image.shape[1], image.shape[0]), (0, 0, 0))

    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')

    black_image.paste(segmented_image, mask=transparency_mask_image)

    return black_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="hard", help="Prompt type: hard/soft/ensemble")
    args = parser.parse_args()

    first_label = "doctor" #input("Enter the first label: ")
    second_label = "skilled doctor" #input("Enter the second label: ")

    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint='/work/tesi_aonori/models/model.pth').to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    image_path = "images/doctors.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    cropped_boxes = []

    for mask in masks:
        cropped_boxes.append(segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))

    if args.prompt == "hard":
        first_prompt = 'A man.' #f'A photo of a {first_label}.'
        second_prompt = 'A woman.'#f'A photo of a {second_label}.'
        prompts = [first_prompt, second_prompt]

        tokenized_prompts = torch.cat([clip.tokenize(prompt) for prompt in prompts]).to(device)

        with torch.no_grad():
            text_features = model.encode_text(tokenized_prompts)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            scores = [0, 0]
            for i, text_feature in enumerate(text_features):
                scores[i] = retriev(cropped_boxes, text_feature)

                #TODO: threshold forse non bellissima, dipende da quanti elementi ci sono in totale, per via della softmax
                indixes = get_indixes_of_values_above_threshold(scores[i], 0.10)

                # Keep only indixes in the top 10 of values above mean value
                # values, idxs = scores[i].topk(10)
                # mean_value= values.mean()
                # indixes = [i for i, v in enumerate(values) if v > mean_value]

                segmentation_masks = []

                for seg_idx in indixes:
                    segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
                    segmentation_masks.append(segmentation_mask_image)

                original_image = Image.open(image_path)
                overlay_image = Image.new('RGBA', (image.shape[1],image.shape[0]), (0, 0, 0, 0))
                overlay_color = (255, 0, 0, 200)

                draw = ImageDraw.Draw(overlay_image)
                for segmentation_mask_image in segmentation_masks:
                    draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

                result_image = Image.alpha_composite(original_image.convert('RGBA'), overlay_image)

                result_image.save(f'images/segmentation/{args.prompt}/result_{i}.png')

    elif args.prompt == "soft":
        prompt = "soft"
    elif args.prompt == "ensemble":
        prompt = "ensemble"
    else:
        raise ValueError("Prompt type must be 'hard', 'soft' or 'ensemble'")

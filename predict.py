import argparse
import json

from util import process_image, load_model

import torch


def predict():
    args = cli()

    device = torch.device("cuda" if args.gpu else "cpu")
    print(f'Device: {device}')
    
    image = process_image(args.image_path)
    model = load_model()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    top_ps, top_class = _predict(model, image, args.top_k, device)

    for i, c in enumerate(top_class):
        print(f"Prediction {i + 1}: "
              f"{cat_to_name[c]} .. "
              f"({100.0 * top_ps[i]:.3f}%)")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("checkpoint")
    parser.add_argument("--top_k", default=1, type=int)
    parser.add_argument("--category_names",
                        default="cat_to_name.json")
    parser.add_argument("--gpu", action="store_true")

    return parser.parse_args()


def _predict(model, image, topk, device):
    tensor_image = torch.from_numpy(image).type(torch.FloatTensor)
    tensor_image = tensor_image.unsqueeze_(0)
    
    model.to(device)
    model.eval()
        
    with torch.no_grad():
        print(f'Device: {device}')
        output = model(tensor_image.to(device)).cpu()
        
        probs = torch.exp(output)
        top_p, top_class = probs.topk(topk, dim=1)
        
        top_p = top_p.numpy()[0]
        top_class = top_class.numpy()[0]

    idx_to_class = {val: key for key, val in
                    model.class_to_idx.items()}
    top_class = [idx_to_class[i] for i in top_class]

    return top_p, top_class


if __name__ == "__main__":
    predict()

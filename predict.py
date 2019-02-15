import torch
import json
import predict_f as pred
import utils
import argparse
# NOTE: a prod-ready app could also use path validation and range checks

parser = argparse.ArgumentParser(description='Network parameters')

parser.add_argument('image',
        help='path to the image that will be classified')

parser.add_argument('checkpoint',
        help='path to a file with a saved model')
parser.add_argument('--top_k',
        type=int,
        help='Amount of top classes to show',
        default=1)
parser.add_argument('--category_names',
        help='path to a file with category names',
        default='./cat_to_name.json')
parser.add_argument('--gpu',
        help='if specified, will use a GPU for training, if available',
        action="store_true",
        default=False)

args = parser.parse_args()

path = utils.get_checkpoints_path(args.checkpoint)
model = utils.get_saved_model(path)

probs, classes = pred.predict(args.image, model, args.top_k)

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

checkpoint = torch.load(path)

print(checkpoint['epochs'])
idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}

for i in range(0, len(probs)):
    print(probs[i], classes[i], idx_to_class.get(classes[i]), cat_to_name.get(str(classes[i])))

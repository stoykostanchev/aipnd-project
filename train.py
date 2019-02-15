import argparse
import json
import train_f as train


parser = argparse.ArgumentParser(description='Network parameters')

parser.add_argument('data_directory',
        help='path to the directory where data can be found. The directory needs to contain 3 subfolders - ./train, ./valid and ./test, and inside of them - entities, structured in a way that the ImageFolder loader can process')

parser.add_argument('--save_dir',
        help='path to where any checkpoints will be stored',
        default='./')
parser.add_argument('--arch',
        choices=['vgg19', 'vgg12'],
        help='network architecture',
        default='vgg19')
parser.add_argument('--learning_rate',
        help='learning rate. Must be between 0 and 1',
        default=0.01,
        type=float)
parser.add_argument('--hidden_units',
        help='number of hidden nodes in the classifier layer',
        default=4096,
        type=int)
parser.add_argument('--epochs',
        help='number of times to iterate data. Higher numbers might lead to overfitting.',
        default=7,
        type=int)
parser.add_argument('--gpu',
        help='if specified, will use a GPU for training, if available',
        default=False)

args = parser.parse_args()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

train.train(args.data_directory, cat_to_name,
        args.save_dir, args.learning_rate, args.arch,
        args.epochs, args.hidden_units, args.gpu)


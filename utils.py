import torch
import torchvision
from collections import OrderedDict
import data_utils

def get_device(gpu = None):
    '''Uses CPU only if told explicitly or if GPU not available
    '''
    force_cpu = gpu == False or not torch.cuda.is_available()

    return torch.device("cpu" if force_cpu else "cuda:0" )

def get_checkpoints_path(path = None):
    if path == None:
        return './state.pt'
    return path

def get_custom_classifier(model, hidden_units, classes):
    # ======== Define customer classifier ===
    output_classes_count = classes
    input_features = model.classifier[0].in_features # vgg19 = 25088

    return torch.nn.Sequential(OrderedDict([
      ('fc1', torch.nn.Linear(input_features, hidden_units)),
      ('relu', torch.nn.ReLU()),
      ('dropout', torch.nn.Dropout(0.4)),
      ('fc2', torch.nn.Linear(hidden_units, output_classes_count)), #102 classes
      ('output', torch.nn.LogSoftmax(dim=1)) # can be linear isntead
    ]))

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def save_checkpoint(model, epochs, optimizer, cat_to_name, class_to_idx, path=None):
    print('Saving model')

    torch.save({
        'class_to_idx': class_to_idx,
        'cat_to_name' : cat_to_name,
        'epochs': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # The last layer is a transform, we can get the count of outputs from the one before it
        'classes': model.classifier[len(model.classifier) - 2].out_features
    }, get_checkpoints_path(path))

model_cached = None
def get_model(arch, hidden_units, classes):
    global model_cached
    if model_cached != None:
        return model_cached
    
    model = freeze_params(torchvision.models.vgg19(pretrained='imagenet'))
    model.classifier = get_custom_classifier(model, hidden_units, classes)
    model_cached = model
    return model

def get_saved_model(checkpoint_path=None, arch='vgg19', hidden_units=4096, gpu=None, data_dir=None):
    if data_dir == None:
        data_dir = 'flowers' # this DS has 102 classes 


    try:
        device = get_device(gpu)
        checkpoint = torch.load(get_checkpoints_path(checkpoint_path), map_location=str(device))
        print('- Loaded model from a checkpoint -')
        model = get_model(arch, hidden_units, checkpoint['classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
    except:    
        _, image_datasets, _ = data_utils.get_data(data_dir)
        model = get_model(arch, hidden_units, len(image_datasets["test"].classes))
        print('- Could not load previous model -')

    return model


def evaluate_model(dl, model, criterion = None):
    print('- Calculating accuracy and loss ... -')
    correct = 0
    total = 0
    running_loss = 0
    device = get_device()

    with torch.no_grad():
        for ii, (images, labels) in enumerate(dl):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if criterion != None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()

    return 100* correct / total, running_loss / total
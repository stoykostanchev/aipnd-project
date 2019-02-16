import torch
import torchvision
import data_utils
from utils import get_saved_model, get_device, save_checkpoint, get_checkpoints_path, evaluate_model

def train(data_dir, cat_to_name, save_dir=None, lr=0.001, arch='vgg19', max_epochs=0, hidden_units=4096, gpu=None):
    model = get_saved_model(save_dir + 'state.pt', arch, hidden_units, gpu, data_dir)
    dataloaders, image_datasets, _ = data_utils.get_data(data_dir)
    criterion = torch.nn.NLLLoss()

    device = get_device(gpu)
    model.to(device)
    if hasattr(model, 'classifier'):
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    try:
        checkpoint = torch.load(get_checkpoints_path(save_dir + 'state.pt'), map_location=str(device))
        print('- Continuing training from a previous state -')
        epochs = checkpoint['epochs']
        state_dict = checkpoint['optimizer_state_dict']
        optimizer.load_state_dict(state_dict)
    except:
        epochs = 0
        print('- No previous optim data / error during optim setup from checkpoint -')


    print('- Starting from epoch:', epochs)
    print('- End epoch:', max_epochs)

    print_every = 50
    steps = 0
    running_loss = 0
    correct = 0
    total = 0

    print('- Training ... -')

    for e in range(max_epochs):
        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            if e < epochs:
                # Skipping data that we've already used and trained upon
                continue

            steps += 1        
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model.forward(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)        
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if steps % print_every == 0:
                print(
                      "Epoch: {}/{} ... ".format(e+1, max_epochs),
                      "Steps: {} ...".format(steps),
                      "Train loss: {:.4f} ...".format(running_loss/print_every),
                      "Train accuracy: {0:.0%}".format( correct / total )
                     )
                running_loss = 0
                correct = 0
                total = 0
        
        model.eval()
        print("Evaluating epoch {} ... ".format(e+1))
        acc, loss = evaluate_model(dataloaders['valid'], model, criterion)
        print(
          "Validation loss: {:.4f} ...".format(loss),
          "Validation accuracy: %d %%" % (acc)
        )
        model.train()
            
        if  e >= epochs:
            save_checkpoint(model, e, optimizer, cat_to_name, image_datasets['train'].class_to_idx, arch=arch)

    print("Training complete")

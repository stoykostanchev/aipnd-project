global str
from PIL import Image
import numpy as np
import torch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)

    # --- Resize to a min of 255px on a side, keep aspect ratio
    width, height = pil_image.size
    
    if width < height:
        portait = True
        size = 255, round(height / (width / 255))
    else:
        portrait = False
        size = round(width / (height / 255)), 255
        
    pil_image = pil_image.resize(size)
    # ---------
    
    # --- Crop to 224px at the center
    new_width, new_height = pil_image.size

    center_x = round(new_width/2)
    center_y = round(new_height/2)
    desired_size = 224, 224
    desired_width_diff, desired_height_diff = round(desired_size[0]/2), round(desired_size[1]/2)
    left = center_x - desired_width_diff
    right = center_x + desired_width_diff
    top = center_y + desired_height_diff
    bottom = center_y - desired_height_diff
    crop_box = (
        left,bottom,
        right,top
    )      
    pil_image = pil_image.crop(crop_box)
    # ---------
    np_image = np.array(pil_image)
    mean = np.array([0.485, 0.456, 0.406])
    st_dev = np.array([0.229, 0.224, 0.225])
    new_img = (np_image / 255 - mean ) / st_dev
    
    new_img = new_img.transpose(2, 0, 1)
    return new_img


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = np.array([process_image(image_path)])

    with torch.no_grad():
        out = model(torch.from_numpy(img).float())
        exp = torch.exp(out)
        probs, classes = exp.topk(topk)

        return probs.numpy()[0], classes.numpy()[0]

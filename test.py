


from collections import defaultdict
import copy
import random
import os
import shutil
from urllib.request import urlretrieve


import cv2
import matplotlib.pyplot as plt
import numpy as np
# import ternausnet.models
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader

cudnn.benchmark = True

from ednfc import*
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from skimage.io import imsave
from torch.autograd import Variable
from tqdm import tqdm
import torchmetrics


print(torch.__version__)

def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[(mask == 255.0)] = 1.0
    return mask

torch.cuda.set_device(0)

 
class NEPCDataset(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        mask_image_filename=image_filename.split('.jpg')[0]+'_segmentation.png'
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            os.path.join(self.masks_directory, mask_image_filename.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED,
        )
        mask = preprocess_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask, image_filename



val_images_directory='./ISIC-2017_Test_v2_Data'
val_masks_directory='./ISIC-2017_Test_v2_Part1_GroundTruth'
val_images_filenames=os.listdir('./ISIC-2017_Test_v2_Data')
val_transform = A.Compose(
    [
    A.Resize(256, 256), 
     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)
idx=random.sample(range(1, 600), 20)
val_images_filenames_=np.array(val_images_filenames)
val_dataset = NEPCDataset(val_images_filenames_, val_images_directory, val_masks_directory, transform=val_transform,)
print(f'val dataset images: {val_dataset.__len__()} ')


val_dataset_loader=DataLoader(val_dataset,batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

if  os.path.exists('./isic_predictions/ednfc'):
    shutil.rmtree('./isic_predictions/ednfc')

os.makedirs('./isic_predictions/ednfc')


softmax=m = nn.Softmax(dim=1)
loss_fn=torch.nn.CrossEntropyLoss().cuda()
def predict(model):
    model.eval()
    val_dice_list=[]
    val_running_loss=0.0

    for step, (images, labels, image_filename) in enumerate((val_dataset_loader)):
        with torch.no_grad():
            images = Variable(images.cuda(),requires_grad=True)
            labels = Variable(labels.long().cuda(),requires_grad=False)

            with torch.cuda.amp.autocast():
                    output=model(images)
                    output=output.squeeze(1)
                    loss = loss_fn(output, labels)
            
            val_running_loss += loss.item()
            output=torch.argmax(softmax(output), dim=1)
            val_dice_list.append(torchmetrics.functional.dice(output,labels, num_classes=3,average=None, zero_division=1e-6 )[:-1].data.cpu().numpy())                 
            rgb_mask = get_coloured_mask(output.data.cpu().numpy()[0])
            rgb_mask_gt = get_coloured_mask(labels.data.cpu().numpy()[0])
            images=cv2.imread(f'{val_images_directory}/{image_filename[0]}')
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            images=cv2.resize(images, (256,256), interpolation = cv2.INTER_CUBIC)
            img = cv2.addWeighted(images, 1, rgb_mask, 0.3, 0)
            img_gt = cv2.addWeighted(images, 1, rgb_mask_gt, 0.3, 0)
            imsave(f'./isic_predictions/ednfc/prediction_{image_filename[0]}', img)
            imsave(f'./isic_predictions/ednfc/gt_{image_filename[0]}', img_gt)
            imsave(f'./isic_predictions/ednfc/original_{image_filename[0]}', images) 
    val_dice_list=np.array(val_dice_list)
    print(f'Val Dice: {np.mean(val_dice_list, axis=0)} Val Loss: {val_running_loss/(step+1)}')
    print('-' * 70)
    return np.mean(val_dice_list, axis=0)


model=torch.load(f'ednfc_best.pt') 
predictions = predict(model.cuda())








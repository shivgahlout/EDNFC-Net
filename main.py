import copy
import random
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from ednfc import*
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from torch.autograd import Variable
from tqdm import tqdm
import torchmetrics
from albumentations import*


""" 
# uncomment to track logs with wandb
import wandb
wandb.login()
wandb.init(name='ISIC Dataset', 
           project='EDNFCSegmentation') 
 """

import argparse

parser = argparse.ArgumentParser(description='Model Training')

parser.add_argument('--gpu_no', type=int, help='gpu-name', default=0)
parser.add_argument('--epochs', type=int, help='no of epochs', default=100)
parser.add_argument('--lr', type=int, help='learning rate', default=5e-3)
parser.add_argument('--in_channels', type=int, help='input channels of EDNFC-Net', default=3)
parser.add_argument('--out_channels', type=int, help='output channels of first FCC in EDNFC-Net', default=64)
parser.add_argument('--bottleneck_channels', type=int, help='inter-FCC bottleneck in EDNFC-Net', default=16)
parser.add_argument('--n_classes', type=int, help='no of output classes in EDNFC-Net', default=2)
args = parser.parse_args()


print(torch.__version__)

def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[(mask == 255.0)] = 1.0
    return mask


def display_image_grid(images_filenames, images_directory, masks_directory, predicted_masks=None):
    cols = 3 if predicted_masks is not None else 2
    rows = len(images_filenames)
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(5, 35))
    for i, image_filename in enumerate(images_filenames):
        image = cv2.imread(os.path.join(images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(masks_directory, image_filename.split('.jpg')[0]+'_segmentation.png'), cv2.IMREAD_UNCHANGED,)
        mask = preprocess_mask(mask)
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")

        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("GT: {}".format(np.unique(mask)))

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

        if predicted_masks is not None:
            predicted_mask = predicted_masks[i]
            ax[i, 2].imshow(predicted_mask, interpolation="nearest")
            ax[i, 2].set_title(f"Predicted mask: {np.unique(predicted_mask)}")
            ax[i, 2].set_axis_off()
    plt.tight_layout()
    plt.show()
 
images_directory='./ISIC-2017_Training_Data'
masks_directory='./ISIC-2017_Training_Part1_GroundTruth'
test_images_filenames=os.listdir('./ISIC-2017_Training_Data')
idx=idx=random.sample(range(1, 2000), 20)

display_image_grid(np.array(test_images_filenames)[idx], images_directory, masks_directory)

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
        return image, mask



train_transform = A.Compose(
    [
        A.Resize(256, 256),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.OneOf([
        HueSaturationValue(20,25,20),
        CLAHE(clip_limit=2),           
    ], p=0.3),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

train_images_directory='./ISIC-2017_Training_Data'
train_masks_directory='./ISIC-2017_Training_Part1_GroundTruth'
train_images_filenames=os.listdir('./ISIC-2017_Training_Data')
train_dataset = NEPCDataset(train_images_filenames, train_images_directory, train_masks_directory, transform=train_transform,)

print(f'train dataset images: {train_dataset.__len__()} ')

val_images_directory='./ISIC-2017_Validation_Data'
val_masks_directory='./ISIC-2017_Validation_Part1_GroundTruth'
val_images_filenames=os.listdir('./ISIC-2017_Validation_Data')
val_transform = A.Compose(
    [
    A.Resize(256, 256), 
     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)
val_dataset = NEPCDataset(val_images_filenames, val_images_directory, val_masks_directory, transform=val_transform,)
print(f'val dataset images: {val_dataset.__len__()} ')

train_dataset_loader=DataLoader(train_dataset,batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_dataset_loader=DataLoader(val_dataset,batch_size=8, shuffle=False, num_workers=2, pin_memory=True)



def visualize_augmentations(dataset, idx=0, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    for i in range(samples):
        image, mask = dataset[idx]
        mask[mask==255]=3.0
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask: {}".format(np.unique(mask)))
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()




visualize_augmentations(train_dataset, idx=545)



torch.cuda.set_device(args.gpu_no)


model=EDNFC(in_channels=args.in_channels, out_channels=args.out_channels, \
    bottleneck_channels=args.bottleneck_channels, \
    n_classes=args.n_classes).cuda()   



num_epochs=args.epochs
scaler = torch.cuda.amp.GradScaler()
softmax = nn.Softmax(dim=1)
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 1e-6)

best_dsc=0.0
for epoch in range(num_epochs):
    model.train()
    dice_list=[]
    train_running_loss=0.0

    for step, (images, labels) in enumerate((train_dataset_loader)):
            images = Variable(images.cuda(args.gpu_no),requires_grad=True)
            labels = Variable(labels.long().cuda(args.gpu_no),requires_grad=False)

            with torch.cuda.amp.autocast():
                    output=model(images)
                    output=output.squeeze(1)
                    loss = loss_fn(output, labels)

            train_running_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            output=torch.argmax(softmax(output), dim=1)
            dice_list.append(torchmetrics.functional.dice(output,labels, num_classes=args.n_classes+1,average=None, zero_division=1e-6 )[:-1].data.cpu().numpy())
    
    dice_list=np.array(dice_list)
    print(f'Epoch: {epoch} Train Dice: {np.mean(dice_list, axis=0)} Train Loss: {train_running_loss/(step+1)}')
    print('*' * 70)
    lr_scheduler.step()
    
    model.eval()
    val_dice_list=[]
    val_running_loss=0.0

    for step, (images, labels) in enumerate((val_dataset_loader)):
        with torch.no_grad():
            images = Variable(images.cuda(args.gpu_no),requires_grad=True)
            labels = Variable(labels.long().cuda(args.gpu_no),requires_grad=False)

            with torch.cuda.amp.autocast():
                    output=model(images)
                    output=output.squeeze(1)
                    loss = loss_fn(output, labels)

            val_running_loss += loss.item()
            output=torch.argmax(softmax(output), dim=1)

            val_dice_list.append(torchmetrics.functional.dice(output,labels, num_classes=args.n_classes+1,average=None, zero_division=1e-6 )[:-1].data.cpu().numpy())
   
    val_dice_list=np.array(val_dice_list)
    print(f'Epoch: {epoch} Val Dice: {np.mean(val_dice_list, axis=0)} Val Loss: {val_running_loss/(step+1)}')

    """ 
    # uncomment to track logs with wandb
    wandb.log({
        "Train Loss": train_running_loss/(step+1),
        "Train DSC": np.mean(dice_list, axis=0)[1],
        "Valid Loss": val_running_loss/(step+1),
        "Valid DSC": np.mean(val_dice_list, axis=0)[1]}) """

    if np.mean(val_dice_list, axis=0)[1]>best_dsc:
        print(f'saving best model with dice  at epoch {epoch}')
        torch.save(model, 'ednfc_best.pt')
        best_dsc=np.mean(val_dice_list, axis=0)[1]
        best_epoch=epoch
    print(f'Best epoch {best_epoch}; best DSC {best_dsc}')
    print('-' * 70)
torch.save(model, 'ednfc_final.pt')    



from ednfc_utils import*


""" # uncomment to track logs with wandb
import wandb
wandb.login()
wandb.init(name='ISIC Dataset', 
           project='EDNFCSegmentation') """ 


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




images_directory='./ISIC-2017_Training_Data'
masks_directory='./ISIC-2017_Training_Part1_GroundTruth'
test_images_filenames=os.listdir('./ISIC-2017_Training_Data')
idx=idx=random.sample(range(1, 2000), 20)
display_image_grid(np.array(test_images_filenames)[idx], images_directory, masks_directory)



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
    

    train_losses = AverageMeter('train_loss', ':.5f')
    train_DSC = AverageMeter('train_dsc', ':5f')

    val_losses = AverageMeter('val_loss', ':.5f')
    val_DSC = AverageMeter('val_dsc', ':5f')

    progress = ProgressMeter(
        len(train_dataset_loader),
        [ train_losses, train_DSC, val_losses, val_DSC],
        prefix="Epoch: [{}]".format(epoch))
    model.train()
    dice_list=[]
    train_running_loss=0.0
   
    for step, (images, labels,_) in enumerate((train_dataset_loader)):
            
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
            dsc=torchmetrics.functional.dice(output,labels, num_classes=args.n_classes+1,average=None, zero_division=1e-6 )[:-1]
            train_losses.update(loss.item(), images.size(0))
            train_DSC.update(dsc[1].data.cpu().numpy(), images.size(0))
         
            
    dice_list=np.array(dice_list)
    lr_scheduler.step()
    
    
    model.eval()
    val_dice_list=[]
    val_running_loss=0.0

    for step, (images, labels,_) in enumerate((val_dataset_loader)):
        with torch.no_grad():
            images = Variable(images.cuda(args.gpu_no),requires_grad=True)
            labels = Variable(labels.long().cuda(args.gpu_no),requires_grad=False)

            with torch.cuda.amp.autocast():
                    output=model(images)
                    output=output.squeeze(1)
                    loss = loss_fn(output, labels)

            val_running_loss += loss.item()
            output=torch.argmax(softmax(output), dim=1)
            dsc=torchmetrics.functional.dice(output,labels, num_classes=args.n_classes+1,average=None, zero_division=1e-6 )[:-1]
            val_losses.update(loss.item(), images.size(0))
            val_DSC.update(dsc[1].data.cpu().numpy(), images.size(0))
    
    
    
    """ # uncomment to track logs with wandb
    wandb.log({
        "Train Loss": train_losses.avg,
        "Train DSC": train_DSC.avg,
        "Valid Loss": val_losses.avg,
        "Valid DSC": val_DSC.avg}) """

    progress.display(epoch + 1)
    if val_DSC.avg>best_dsc:
        print(f'saving best model with dice  at epoch {epoch}')
        torch.save(model, 'ednfc_best.pt')
        best_dsc=val_DSC.avg
        best_epoch=epoch
    print(f'Best epoch {best_epoch}; best DSC {best_dsc}')
    # print('-' * 70)
torch.save(model, 'ednfc_final.pt')    



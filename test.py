from ednfc_utils import*

print(torch.__version__)


torch.cuda.set_device(0)

 

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



if  os.path.exists('./isic_predictions/ednfc'):
    shutil.rmtree('./isic_predictions/ednfc')

os.makedirs('./isic_predictions/ednfc')


softmax = nn.Softmax(dim=1)
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








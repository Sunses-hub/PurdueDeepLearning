import os 
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tvt
import random
from PIL import Image

class MyCOCODataset(Dataset):
    def __init__(self, folder_name, aug=False, train=True):
        super().__init__()
        self.folder_name = folder_name # dataset path
        self.files = os.listdir(folder_name) # file names in list
        random.shuffle(self.files) # shuffle the list
        # Later one can add augmentations to this pipeline easily
        if train:
            self.transforms = tvt.Compose([
                #tvt.RandomHorizontalFlip(p=0.5),
                #tvt.RandomVerticalFlip(p=0.5),
                #tvt.RandomRotation(degrees=15),
                #tvt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                tvt.ToTensor(),
            ])
        else:
            self.transforms = tvt.Compose([
                tvt.ToTensor()
            ])
        # class names
        self.classes = ["airplane", "bus", "cat", "dog", "pizza"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file_name = self.files[item]
        img = Image.open(os.path.join(self.folder_name, file_name)) # load image
        img = self.transforms(img) # convert it to tensor with normalization
        class_name = file_name.split("_")[0] # get the index of label
        label = self.classes.index(class_name) # get label
        return img, label

def plot_images(train_data):
    # class names
    classes = ["airplane", "bus", "cat", "dog", "pizza"]
    # plot random 5 images for each class
    fig, axs = plt.subplots(5, 5)
    fig.tight_layout()
    for i, cat in enumerate(classes):
        files = train_data[cat]
        chosen = random.sample(files, 5) # sample 5 random images
        for j, file_name in enumerate(chosen):
            img = Image.open(os.path.join('train_data', file_name)) # load image
            axs[i, j].imshow(img) # plot image
            axs[i, j].axis('off')
            axs[i, j].set_title(f"{cat} {j + 1}", size=7)
    plt.show()

def set_datasets(data_dir="/Users/berksahin/Desktop"):
    # create folders
    if not os.path.exists("train_data"):
        os.mkdir("train_data")
    if not os.path.exists("val_data"):
        os.mkdir("val_data")

    annFile = os.path.join(data_dir, "coco/annotations/instances_train2014.json")
    coco = COCO(annFile)
    # classes for the problem in hw4
    classes = ['airplane', 'bus', 'cat', 'dog', 'pizza']
    catIds = coco.getCatIds(catNms=classes)  # get class indices
    # keys: class names, values: list of files names
    train_data = dict(zip(classes, [[] for i in range(len(classes))]))
    val_data = dict(zip(classes, [[] for i in range(len(classes))]))
    print("dataset generation has started...")
    for i, idx in enumerate(catIds):
        imgIds = coco.getImgIds(catIds=idx)
        imgIds = np.random.choice(imgIds, size=2000, replace=False)

        for counter, img_idx in enumerate(imgIds):
            # get the file name of the image
            file_name = coco.loadImgs(int(img_idx))[0]['file_name']
            img_path = os.path.join(data_dir, f"coco/images/{file_name}")
            # open the image as PIL image in RGB format
            img = Image.open(img_path).convert("RGB")
            img = img.resize((64, 64)) # resize

            class_name = classes[i]
            save_name = class_name + "_" + file_name
            if counter < 1500:
                save_dir = "train_data"
                train_data[class_name].append(save_name)
            else:
                save_dir = "val_data"
                val_data[class_name].append(save_name)
            # save the image in new dataset folder
            img.save(os.path.join(save_dir, save_name))

    print("train and validation datasets are ready!")

    return {"train_data" : train_data, "val_data" : val_data}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/Users/berksahin/Desktop", help="Current directory path of the coco folder")
    args = parser.parse_args()

    np.random.seed(5)
    random.seed(5)
    # create the dataset 
    data = set_datasets(args.data_dir)
    plot_images(data["train_data"])

    # construct datasets
    train_dataset = MyCOCODataset(folder_name='train_data')
    val_dataset = MyCOCODataset(folder_name='val_data')

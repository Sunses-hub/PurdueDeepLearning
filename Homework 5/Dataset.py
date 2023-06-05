
from pycocotools.coco import COCO
import os
import argparse
import skimage
from skimage import io
from skimage.transform import resize
import numpy as np
from torch import nn
import random
import matplotlib.pyplot as plt
import cv2
import pickle
import torchvision.transforms as tvt
import torch

class MyCOCODataset(nn.Module):
    def __init__(self, data, label_map, train=True):
        super(MyCOCODataset, self).__init__()
        self.files = list(data.keys())
        self.data = data
        self.label_map = label_map
        # path of extracting data 
        if train:
            self.path = 'train_data'
        else:
            self.path = 'test_data'

        self.transforms = tvt.Compose([tvt.ToTensor()
                                       #tvt.RandomCrop(224,224)])
                                      ])
    def __len__(self):
        return len(self.files) # data size 

    def __getitem__(self, item):
        img_file = self.files[item]
        # get the image with transformations
        img = io.imread(os.path.join(self.path, img_file))
        #img = Image.open(os.path.join(self.path, img_file))
        img = self.transforms(img)
        # get the label
        label = self.label_map[self.data[img_file]['category_id']]
        # get the bounding box
        [x1, y1, w, h] = self.data[img_file]['bbox']
        x2, y2 = x1+w, y1+h # lower right corner 
        return {"image": img,
                "label": torch.tensor(label),
                "bbox": torch.tensor([x1/255, y1/255, x2/255, y2/255])}




# dataset generator for train and validation sets
def data_generator(coco, catIds, data_path, train=True):

    # keeps the file names as keys and annotations as values
    data = {}

    for cat_id in catIds:
        imgIds = coco.getImgIds(catIds=cat_id)
        for img_id in imgIds:
            # get annotations
            annIds = coco.getAnnIds(imgIds=img_id, catIds=cat_id, iscrowd=False,
                                    areaRng=[200 * 200, float('inf')])
            # load annotations
            anns = coco.loadAnns(annIds)
            # use only images with one dominant bbox
            if len(anns) != 1:
                continue
            ann = anns[0]

            # read the image and resize
            img = coco.loadImgs(img_id)[0]
            file_name = img['coco_url'].split('/')[-1]
            I = io.imread(os.path.join(data_path, file_name))
            if len(I.shape) == 2:
                I = skimage.color.gray2rgb(I)
            # retrieve image width and height 
            img_h, img_w = I.shape[0], I.shape[1]
            I = resize(I, (256, 256), anti_aliasing=True, preserve_range=True)
            image = np.uint8(I) # image format [0, 255]
            # save the image
            if train:
                io.imsave(os.path.join("train_data", file_name), image)
            else:
                io.imsave(os.path.join("test_data", file_name), image)

            # scale annotations (bounding boxes)
            x_scale, y_scale = 256 / img_h, 256 / img_w
            [x, y, w, h] = ann['bbox']
            ann['bbox'] = [x * y_scale, y * x_scale, w * y_scale, h * x_scale]
            # delete unnecessary elements
            del ann['segmentation'], ann["iscrowd"]
            # save annotations with its image to a train dict.
            data[file_name] = ann

    return data

def plot_images(class_list, coco_inv_labels, data, catIds, img_nmb=5):
    # initialize counter and image dictionaries
    counter = {ids: 0 for ids in catIds}
    imgs = {ids: [] for ids in catIds}
    # choose img_nmb (5) elements from each category
    file_list = list(data.keys())
    random.seed(6)
    random.shuffle(file_list)

    for file_name in file_list:
        cat_id = data[file_name]['category_id']
        if counter[cat_id] < img_nmb:
            counter[cat_id] += 1
            imgs[cat_id].append(file_name)
    # plot them
    fig, axs = plt.subplots(3, img_nmb)
    fig.tight_layout()

    for row, cat in enumerate(list(imgs.keys())):
        for col, file_name in enumerate(imgs[cat]):
            class_name = class_list[coco_inv_labels[cat]]
            # load annotations
            [x, y, w, h] = data[file_name]['bbox']
            # load image
            I = io.imread(os.path.join("train_data", file_name))
            image = np.uint8(I)
            # draw bounding box
            image = cv2.rectangle(image, (round(x), round(y)),
                                  (round(x + w), round(y + h)),
                                  (36, 256, 12), 2)
            # plot the image
            axs[row, col].imshow(image)
            axs[row, col].axis('off')
            axs[row, col].set_title(f"{class_name} {col + 1}", size=7)
            if class_name == "pizza":
                print(file_name)

    plt.show()

# test code 
if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dir", default="/Users/berksahin/Desktop",
                        help="parent directory of coco dataset")

    args = parser.parse_args()
    # important directories
    coco_dir = args.coco_dir
    train_dir = os.path.join(coco_dir, "coco/train2014")
    test_dir = os.path.join(coco_dir, "coco/test2014")
    ann_dir = os.path.join(coco_dir, "coco/annotations2014")

    class_list = ['bus', 'cat', 'pizza']
    train_json = 'instances_train2014.json'
    test_json = 'instances_val2014.json'

    if not os.path.exists("train_data"):
        os.mkdir("train_data")
    if not os.path.exists("test_data"):
        os.mkdir("test_data")

    # train and test COCOs
    coco_train = COCO(os.path.join(ann_dir, train_json))
    coco_test = COCO(os.path.join(ann_dir, test_json))
    # mapping from coco labels to my labels
    coco_inv_labels = {}
    catIds = coco_train.getCatIds(catNms=class_list)
    for idx, catId in enumerate(sorted(catIds)):
        coco_inv_labels[catId] = idx

    # save inverse label map
    with open('inv_map.pkl', 'wb') as f:
        pickle.dump(coco_inv_labels, f)
        print("Inverse map saved.")

    print("datasets are being generated...")
    train_data = data_generator(coco_train, catIds, train_dir, train=True)
    val_data = data_generator(coco_test, catIds, test_dir, train=False)
    print("datasets were generated. ")
    plot_images(class_list, coco_inv_labels, train_data, catIds)
    print("sample images are plotted.")
    # save dictionaries to be used later
    with open('train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
        print("train data saved successfully to file")
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(val_data, f)
        print("test data saved successfully to file")
    # read train dictionary/data
    with open('train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    # test train dataset
    data = MyCOCODataset(train_data, coco_inv_labels)
    print("Length of the dataset:", len(data))
    out = data[0]
    print("image:", out["image"].shape)
    print("label:", out["label"])
    print("bbox:", out["bbox"])
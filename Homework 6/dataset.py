
import torch
from pycocotools.coco import COCO
import pickle
import argparse
import os
import numpy as np
import random
import torch.nn as nn
import cv2
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
import skimage


class YoloDataset(nn.Module):
    def __init__(self, coco, catIds, data_path, coco_inv_labels=None,
                 yolo_interval=20, img_size=256, max_obj=14, anchor_num=5, transform=True,
                 train=True):
        super(YoloDataset, self).__init__()
        # dataset
        self.coco = coco
        self.catIds = catIds
        self.data_path = data_path
        self.train = train
        self.img_size = img_size
        self.coco_inv_labels = coco_inv_labels
        # pre-processing
        self.transform = tvt.Compose([tvt.ToTensor()]) if (transform != None) else None
        # yolo parameters
        self.yolo_interval = yolo_interval
        self.num_cells_width = self.num_cells_height = img_size // self.yolo_interval
        self.anchor_num = anchor_num
        self.max_obj = max_obj
        # dataset generator
        self.folder_name = "train_data" if train else "test_data"
        self.data = self.data_generator() if not os.path.exists(self.folder_name + ".pkl") \
            else pickle.load(open(self.folder_name + ".pkl", "rb"))
        self.file_list = os.listdir(self.folder_name)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        # get the image
        I = io.imread(os.path.join(self.folder_name, self.file_list[item]))
        image = np.uint8(I)
        if self.transform != None:
            image = self.transform(image)
        # get annotations
        ground_truths = self.data[self.file_list[item]]
        return image, ground_truths

    def yolo_extractor(self, anns, x_scale, y_scale):
        yolo_tensor = np.zeros((self.num_cells_width * self.num_cells_height, self.anchor_num, 8)) # you may need to add 9th element later
        # save cell index, anchor num and label
        cell_anc_cat = np.zeros((self.max_obj, 4))
        cell_anc_cat[:,-2:] = 13 # 13 means there is no object
        for i, ann in enumerate(anns):
            # take the bounding box
            class_idx = self.coco_inv_labels[ann['category_id']]
            [x, y, w, h] = ann['bbox']
            # scale the images due to resizing
            [x, y, w, h] = [x * x_scale, y * y_scale, w * x_scale, h * y_scale]
            # bbox center
            x_center, y_center = y + h/2, x + w/2
            # cell index (i, j)
            row_cell_idx = min(x_center // self.yolo_interval, self.num_cells_height - 1) # ith row
            col_cell_idx = min(y_center // self.yolo_interval, self.num_cells_width - 1) # jth column
            # bounding box scale
            bw = w / self.yolo_interval
            bh = h / self.yolo_interval
            # cell center
            cell_i_center = row_cell_idx * self.yolo_interval + self.yolo_interval / 2
            cell_j_center = col_cell_idx * self.yolo_interval + self.yolo_interval / 2
            # calculate difference between centers
            dx = (x_center - cell_i_center) / self.yolo_interval
            dy = (y_center - cell_j_center) / self.yolo_interval
            # aspect ratio
            AR = h / w
            if AR <= 0.2:             anc_idx = 0
            if 0.2 < AR <= 0.5:       anc_idx = 1
            if 0.5 < AR <= 1.5:       anc_idx = 2
            if 1.5 < AR <= 4.0:       anc_idx = 3
            if 4.0 < AR:              anc_idx = 4
            yolo_vector = np.array([1, dx, dy, bh, bw, 0, 0, 0])
            yolo_vector[5 + class_idx] = 1
            # save the yolo_vector to yolo_tensor
            yolo_tensor[int(row_cell_idx * self.num_cells_width + col_cell_idx), anc_idx, :] = yolo_vector
            cell_anc_cat[i,:2] = (row_cell_idx, col_cell_idx)
            cell_anc_cat[i, 2] = anc_idx
            cell_anc_cat[i, 3] = class_idx

        return cell_anc_cat, yolo_tensor

    def data_generator(self):
        # keeps the file names as keys and annotations as values
        data = {}

        for cat_id in catIds:
            imgIds = self.coco.getImgIds(catIds=cat_id)
            for img_id in imgIds:
                # get annotations
                annIds = self.coco.getAnnIds(imgIds=img_id, catIds=cat_id,
                                        iscrowd=False,
                                        areaRng=[64*64, float('inf')])
                # load annotations
                anns = self.coco.loadAnns(annIds)
                if len(anns) < 1:
                    continue

                # load the image with resizing
                img = self.coco.loadImgs(img_id)[0]
                I = io.imread(os.path.join(self.data_path, img['file_name']))
                if len(I.shape) == 2:
                    I = skimage.color.gray2rgb(I)
                img_h, img_w = I.shape[0], I.shape[1]
                I = resize(I, (self.img_size, self.img_size), anti_aliasing=True, preserve_range=True)
                image = np.uint8(I)
                # scale annotation bounding boxes
                x_scale, y_scale = self.img_size / img_w, self.img_size / img_h
                # get yolo_tensor and other annotations
                cell_anc_cat, yolo_tensor = self.yolo_extractor(anns, x_scale, y_scale)
                data[img['file_name']] = {"cell_idx" : cell_anc_cat[:,:2],
                                          "anchor_idx" : cell_anc_cat[:,2],
                                          "label" : cell_anc_cat[:,3],
                                          "yolo_tensor" : yolo_tensor}
                # save the image
                if self.train:
                    io.imsave(os.path.join("train_data", img['file_name']), image)
                else:
                    io.imsave(os.path.join("test_data", img['file_name']), image)

        print("data generation finished and dictionary was saved.")
        # save the data as .pkl
        pickle.dump(data, open(self.folder_name + '.pkl', 'wb'))
        return data

    def plot_images(self, data_loader, sample_num=3):
        class_list = ['bus', 'cat', 'pizza']  # class list

        class_counter = {idx: 0 for idx in range(3)}

        # display the predictions in a figure
        fig, axs = plt.subplots(3, sample_num)
        #fig.tight_layout()

        img_counter = 0
        for k, data in enumerate(data_loader):
            if class_counter[0] == sample_num and class_counter[1] == sample_num and class_counter[2] == sample_num:
                break
            imgs, gts = data
            yolo_tensor, anchor_idx = gts['yolo_tensor'].numpy(), gts['anchor_idx'].numpy()
            cell_idx, labels = gts['cell_idx'].numpy(), gts['label'].numpy()

            yolo_tensor = yolo_tensor.reshape(self.num_cells_height, self.num_cells_width, self.anchor_num, 8)

            # prepare image for display
            img = np.uint8(imgs[0].numpy() * 255)
            img = img.transpose((1, 2, 0))
            img = np.ascontiguousarray(img)

            label = int(labels[0, 0].item())
            if class_counter[label] >= sample_num:
                continue
            else:
                class_counter[label] += 1
            class_name = class_list[label]

            obj_idxs = np.where(anchor_idx[0] != 13)[0]
            # iterate through objects
            for obj_idx in range(len(obj_idxs)):
                # ground truths
                label = int(labels[0, obj_idx].item())

                row_cell_idx, col_cell_idx = cell_idx[0, obj_idx]
                anc_idx = anchor_idx[0, obj_idx]

                [row_cell_idx, col_cell_idx, anc_idx] = list(map(lambda x: int(x.item()), [row_cell_idx, col_cell_idx, anc_idx]))
                # pick the yolo_vector
                yolo_vector = yolo_tensor[row_cell_idx, col_cell_idx, anc_idx]

                # calculate ground truth bbox size
                h = yolo_vector[3].item() * self.yolo_interval
                w = yolo_vector[4].item() * self.yolo_interval
                # calculate cell centers
                cell_i_center = row_cell_idx * self.yolo_interval + self.yolo_interval / 2
                cell_j_center = col_cell_idx * self.yolo_interval + self.yolo_interval / 2
                # calculate the center of gt bbox
                x_center = yolo_vector[1].item() * self.yolo_interval + cell_i_center
                y_center = yolo_vector[2].item() * self.yolo_interval + cell_j_center
                [x1, y1, x2, y2] = [round(y_center - w / 2), round(x_center - h / 2), round(y_center + w / 2),
                                    round(x_center + h / 2)]
                # draw the bounding box
                img = cv2.rectangle(img, (round(x1), round(y1)),
                                    (round(x2), round(y2)),
                                    (36, 256, 12), 2)

                img = cv2.putText(img, class_list[label], (round(x1), round(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.8, (36, 256, 12), 3) #0.8

            # plot the image
            row, col = img_counter // sample_num, img_counter % sample_num
            axs[row, col].imshow(img)
            axs[row, col].axis('off')
            axs[row, col].set_title(f"class: {class_name}", size=8)
            img_counter += 1

        if self.train:
            name = "train_samples"
        else:
            name = "test_samples"
        plt.savefig(f"{name}.jpeg")
        print("Predictions are plotted and figure is saved!")

if __name__ == "__main__":
    # important directories

    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dir", default="/Users/berksahin/Desktop",
                        help="parent directory of coco dataset")

    args = parser.parse_args()

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

    seed = 16 # 7
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmarks = False

    # train and test COCOs
    coco_train = COCO(os.path.join(os.path.join(coco_dir, ann_dir), train_json))
    coco_test = COCO(os.path.join(os.path.join(coco_dir, ann_dir), test_json))
    # mapping from coco labels to my labels
    coco_inv_labels = {}
    catIds = coco_train.getCatIds(catNms=class_list)
    for idx, catId in enumerate(sorted(catIds)):
        coco_inv_labels[catId] = idx

    pickle.dump(coco_inv_labels, open('inv_map.pkl', 'wb'))
    print("Inverse map saved.")

    train_dataset = YoloDataset(coco=coco_train, catIds=catIds, data_path=os.path.join(coco_dir, train_dir),
                                coco_inv_labels=coco_inv_labels, train=True)
    test_dataset = YoloDataset(coco=coco_test, catIds=catIds, data_path=os.path.join(coco_dir, test_dir),
                                coco_inv_labels=coco_inv_labels, train=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print(f"Length of the train dataset: {len(train_dataset)}")

    # plot samples from train dataset
    train_dataset.plot_images(train_loader)
    print("samples from train dataset were saved!")
    # plot samples from test dataset
    test_dataset.plot_images(test_loader)
    print("samples from test dataset were saved!")
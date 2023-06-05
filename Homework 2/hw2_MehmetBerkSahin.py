
# Author: Mehmet Berk Sahin
# Email: sahinm@purdue.edu
# ID: 34740048

# imports
from torch.utils.data import DataLoader
import pandas as pd
import time
from PIL import Image
import numpy as np
import torchvision.transforms as tvt
import random
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader
import os
import torch

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, root):
        super().__init__()
        # meta information
        self.file_list = os.listdir(root)
        if root == "./book_data2":
            self.file_list = [self.file_list[random.randint(0, len(self.file_list)-1)] for i in range(1000)]

        self.root_dir = root
        # transforms
        self.transforms = tvt.Compose([
            tvt.ToTensor(),
            tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            tvt.RandomAffine(degrees=(-90, 90), scale=(0.5, 0.75),
                             translate=(0.1, 0.3)),
            tvt.RandomPerspective(distortion_scale=0.5, p=0.5),
            tvt.ColorJitter(brightness=(0.3, 0.7))
        ])

    def __len__(self):
        # Return the total number of images
        return len(self.file_list)

    def __getitem__(self, index):
        # Return the augmented image at the corresponding index
        img_dir = os.path.join(self.root_dir, self.file_list[index])
        img = Image.open(img_dir)  # load the image
        img_aug = self.transforms(img)
        return (img_aug, random.randint(0, 10))


def cartesian_product(list1, list2):
    result = np.zeros(shape=(len(list1) * len(list2), 3))
    counter = 0
    for i in range(len(list1)):
        for j in range(len(list2)):
            result[counter, 0], result[counter, 1] = list1[i], list2[j]
            counter += 1
    return result



if __name__ == '__main__':

    # DISCLAIMER: DATASET FILE NAMES SHOULD BE "book_data" AND "book_data2"
    # OTHERWISE PROGRAM DOES NOT WORK PROPERLY. THE FORMER ONE INCLUDES OR-
    # IGINAL IMAGES FROM IPHONE. THE SECOND ONE IS DOWNSCALED VERSION.

    print("Packages were imported succesfully.")

    # For reproducibility seed = 0 (taken from weekly slides 2) !TAKEN FROM WEEK 2 SLIDES!
    seed = 3
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmarks = False

    ############ QUESTION 3.2 #################
    # read images as PIL objects
    img = Image.open("StopSign3.jpeg")  # normal image
    cimg = Image.open("StopSign2.jpeg")  # corrupted image (bad angle)
    # convert them to NumPy arrays
    img_np = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    cimg_np = np.array(cimg.getdata()).reshape(cimg.size[1], cimg.size[0], 3)
    # plot the images side-by-side
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("Front View")
    ax[0].axis('off')
    ax[0].imshow(img_np)

    ax[1].set_title("Oblique Angle")
    ax[1].axis('off')
    ax[1].imshow(cimg_np)
    plt.show()

    img = Image.open("StopSign3.jpeg")
    transformer = tvt.Compose([tvt.ToTensor(),
                               tvt.RandomAffine(degrees=0, scale=(1.2, 1.5),
                                                shear=(0, 15)),
                               tvt.ToPILImage()])
    # Affine transform
    aug_img = transformer(img)

    # Perspective transform

    # height, width = aug_img.size[1], aug_img.size[0]
    # height, width = random.randint(width, height), random.randint(width, height)
    # transform = tvt.RandomPerspective()
    # startpoints, endpoints = transform.get_params(height, width, 0.7)
    # print(startpoints)
    # print(endpoints)
    startpoints = [[0, 0], [3052, 0], [3052, 3899], [0, 3899]]
    endpoints = [[843, 1330], [2279, 540], [2505, 3789], [552, 3870]]
    new_img = F.perspective(aug_img, startpoints, endpoints)

    # same plotting procedure
    cimg_np = np.array(cimg.getdata()).reshape(cimg.size[1], cimg.size[0], 3)
    newimg_np = np.array(new_img.getdata()).reshape(new_img.size[1], new_img.size[0], 3)
    # plot the images side-by-side
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    ax[0].imshow(cimg_np)

    ax[1].set_title("Transformed Image")
    ax[1].axis('off')
    ax[1].imshow(newimg_np)
    plt.show()

    # measuring the difference between histograms
    NUM_BINS = 10
    # creating arrays for histograms for each channel
    hist_img = torch.zeros(3, NUM_BINS, dtype=torch.float)
    hist_cimg = torch.zeros(3, NUM_BINS, dtype=torch.float)
    hist_augimg = torch.zeros(3, NUM_BINS, dtype=torch.float)

    pilToTens = tvt.PILToTensor()
    img = pilToTens(img).float()
    cimg = pilToTens(cimg).float()
    aug_img = pilToTens(new_img).float()

    for channel_idx in range(3):
        # stop sign front view
        hist_img[channel_idx] = torch.histc(img[channel_idx],
                                            bins=10, min=0.0, max=255.0)
        hist_img[channel_idx] = hist_img[channel_idx].div(hist_img[channel_idx].sum())

        # stop sign angled view
        hist_cimg[channel_idx] = torch.histc(hist_cimg[channel_idx],
                                             bins=10, min=0.0, max=255.0)
        hist_cimg[channel_idx] = hist_cimg[channel_idx].div(hist_cimg[channel_idx].sum())

        # stop sign augmented
        hist_augimg[channel_idx] = torch.histc(aug_img[channel_idx],
                                               bins=10, min=0.0, max=255.0)
        hist_augimg[channel_idx] = hist_augimg[channel_idx].div(hist_augimg[channel_idx].sum())

    for trial in range(2):
        if trial == 0:
            print("Comparison of front view and oblique angle view")
            other_hist = hist_img
        else:
            print("Comparison of oblique angle view and transformed image")
            other_hist = hist_augimg

        for channel_idx in range(3):
            dist = wasserstein_distance(hist_cimg[channel_idx].cpu().numpy(),
                                        other_hist[channel_idx].cpu().numpy())
            print(f"Wasserstein distance for channel {channel_idx}: {dist}")

    ############ QUESTION 3.3 #################
    root = './book_data'
    # example similar to the example in the assignment
    my_dataset = MyDataset(root)
    print(len(my_dataset))
    index = 4
    print(my_dataset[index][0].shape, my_dataset[index][1])
    index = 9
    print(my_dataset[index][0].shape, my_dataset[index][1])

    # choose three images randomly and display their original and augmented versions
    file_names = os.listdir(root)
    # sample random numbers
    rand_idxs = torch.randint(0, 10, size=(3,))
    # plot configurations
    fig, ax = plt.subplots(3, 2)
    fig.set_size_inches(18.5, 10.5)
    fig.subplots_adjust(wspace=-.75, hspace=0.5)
    # go through random images
    for counter, idx in enumerate(rand_idxs):
        org_img = Image.open(os.path.join(root, file_names[idx]))  # load img
        aug_img = my_dataset[idx][0].numpy()  # take the augmented img as np
        aug_img = np.transpose(aug_img, (1, 2, 0))
        org_img = np.array(org_img.getdata()).reshape(org_img.size[1], org_img.size[0], 3)

        ax[counter, 0].axis('off')
        ax[counter, 0].imshow(org_img)
        ax[counter, 1].axis('off')
        ax[counter, 1].imshow(aug_img)
    plt.suptitle("Original Images vs. Augmented Images")
    plt.show()

    dataset = MyDataset('./book_data')
    dataloader = DataLoader(dataset, batch_size=4, num_workers=2)
    # go through the dataset via dataloader
    for idx, (img, label) in enumerate(dataloader):
        batch_size = img.shape[0]
        fig, ax = plt.subplots(1, batch_size)
        fig.set_size_inches(10, 5)
        fig.suptitle(f"Batch Number {idx+1}")
        # display images in batch
        for idx in range(batch_size):
            img_np = np.transpose(img[idx].numpy(), (1, 2, 0))
            ax[idx].axis('off')
            ax[idx].imshow(img_np)
        plt.show()

    ############ QUESTION 3.4 ############
    root = './book_data2'
    # parameters
    batch_sizes = [5, 10, 15, 20]
    num_wlist = [1, 2, 3, 4, 5]

    dataset = MyDataset(root) # dataset instance
    results = pd.DataFrame(data=cartesian_product(batch_sizes, num_wlist), columns=["batch_size", "num_workers", "time"]) # table for results
    # experiment with the Dataset object
    start_time = time.time()
    for img in dataset:
        pass
    end_time = time.time()
    print("Dataset duration for 1000 images:", end_time - start_time)

    # experiment with the DataLoader object with different parameters
    for row in range(len(results)):
        batch_sizes, num_workers = results.iloc[row, :2]
        batch_sizes, num_workers = int(batch_sizes), int(num_workers)
        dataloader = DataLoader(dataset, batch_size=batch_sizes, num_workers=num_workers)
        print(f"Row Number: {row + 1}/{len(results)}")
        start_time = time.time()
        for idx, img in enumerate(dataloader):
            pass
        end_time = time.time()
        time_passed = np.around(end_time - start_time, 2)
        print(f"DataLoader with batch_size={batch_sizes}, num_workers={num_workers}:", time_passed)
        results.iloc[row, 2] = time_passed

    print("Experiments were completed. Table of the results was constructed and saved. ")
    results.to_csv('results.csv') # save the result

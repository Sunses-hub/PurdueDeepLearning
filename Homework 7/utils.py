
from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
import torch 
from torchvision.utils import save_image
import os


def frechet_value(real_paths, fake_paths, device, dims=2048):
    # load InceptionV3 model 
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    # calculate statistics for both batch 
    m1, s1 = calculate_activation_statistics(real_paths, model, device=device)
    m2, s2 = calculate_activation_statistics(fake_paths, model, device=device)
    # calculate frechet value 
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def calc_frechet(model, dataset, size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # sample real images randomly 
    perm = torch.randperm(len(dataset))
    idxs = perm[:size]
    # img path names
    real_paths = [dataset[i][1] for i in idxs]
    # create fake images 
    z = torch.randn(size, 100, 1, 1)
    fake_imgs = model.generator(z).mul(0.5).add(0.5)
    # create directory for fake images 
    folder_name = "fake_images"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    # save fake images
    fake_paths = []
    for idx in range(fake_imgs.shape[0]):
        f = os.path.join(folder_name, f"fake_img{idx}.jpg")
        save_image(fake_imgs[idx], fp=f)
        fake_paths.append(f)
    # calculate frechet value 
    fid = frechet_value(real_paths, fake_paths, device)
    return fid


# test code
if __name__ == "__main__":
    real_paths = ["pizzas/train/01001.jpg", "pizzas/train/01002.jpg"]
    fake_paths = ["pizzas/train/01003.jpg", "pizzas/train/01004.jpg"]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    fid_val = frechet_value(real_paths, fake_paths, device, dims=2048)
    print(f"frechet value:", fid_val)
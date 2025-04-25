# python imports
import os
import pickle
import hashlib
import urllib
import tarfile
import shutil
import time
from PIL import Image
from tqdm import tqdm
import pickle

# torch imports
import torch
from torch.utils import data
import numpy as np



data_urls = {"data": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"}
data_md5 = "eb9058c3a382ffc7106e4002c42a8d85"


def calculate_md5(fpath, chunk_size=1024*1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()

def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def download_url(url, folder):
    """Download a file from a url and place it in folder.
    Args:
        url (str): URL to download file from
        folder (str): Directory to place downloaded file in
    """
    fpath = os.path.join(os.path.expanduser(folder),
                         os.path.basename(url))

    os.makedirs(os.path.expanduser(folder), exist_ok=True)

    if os.path.exists(fpath):
        return

    try:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(
            url, fpath,
            reporthook=gen_bar_updater()
        )
    except (urllib.error.URLError, IOError) as err:
        print('Failed download.')
        raise err
    return

def extract_targz(src_file, dst_path):
    # create dst folder / extract all files
    print('Extracting ' + src_file + ' to' + dst_path)
    os.makedirs(os.path.expanduser(dst_path), exist_ok=True)
    with tarfile.open(src_file, 'r:gz') as tar:
        tar.extractall(path=dst_path)

class CIFAR100(data.Dataset):
    """
    A simple dataloader for CIFAR100
    """
    def __init__(self,
                 root,
                 label_file=None,
                 num_classes=100,
                 split="train",
                 transform=None):
        assert split in ["train", "val", "test"]
        # root folder, split
        self.root_folder = os.path.join(root, "cifar100")
        self.split = split
        self.transform = transform
        self.n_classes = num_classes

        # download dataset
        if not os.path.exists(os.path.join(self.root_folder, "images", self.split)):
            self._download_dataset(root)
        self.img_label_list = self._load_dataset()

    def _download_dataset(self, data_folder):
        # data folder and data file
        data_folder = os.path.expanduser(data_folder)
        data_file = os.path.join(data_folder,
                                 os.path.basename(data_urls['data']))

        # if we need to download the full dataset
        require_download = True
        if os.path.exists(data_file):
            file_md5 = calculate_md5(data_file)
        else:
            file_md5 = None
        if file_md5 == data_md5:
            require_download = False

        if (not require_download) and \
           os.path.exists(os.path.join(data_folder, 'cifar100-orig')):
           self._parse_pickle_obj(os.path.join(data_folder, 'cifar100-orig'))
        else:
            # corner case: a corrupted file
            if os.path.exists(data_file) and (file_md5 != data_md5):
                print("File corrupted. Remove and re-download ...")
                os.remove(data_file)

            # corner case: the subfolder already exists
            if os.path.exists(os.path.join(data_folder, 'cifar100')):
                shutil.rmtree(os.path.join(data_folder, 'cifar100'))
            if os.path.exists(os.path.join(data_folder, 'cifar100-orig')):
                shutil.rmtree(os.path.join(data_folder, 'cifar100-orig'))

            # download and extract the tar.gz file
            download_url(data_urls['data'], data_folder)
            extract_targz(data_file, data_folder)

            # setup the folders
            print("Setting up data folders ...")
            shutil.move(os.path.join(data_folder, 'cifar-100-python'),
                        os.path.join(data_folder, 'cifar100-orig'))

            self._parse_pickle_obj(os.path.join(data_folder, 'cifar100-orig'))
        return

    def _parse_pickle_obj(self, data_dir):
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        
        def save_array_to_txt(array, filename):
            with open(filename, 'w') as file:
                for item in array:
                    file.write(str(item) + '\n')
        
        file_ = os.path.join(data_dir, self.split)

        data = unpickle(file_)
        images = data[b'data']
        labels = data[b'fine_labels']

        images_r = images[:, :1024].reshape((images.shape[0], 32, 32))
        images_g = images[:, 1024:2048].reshape((images.shape[0], 32, 32))
        images_b = images[:, 2048:].reshape((images.shape[0], 32, 32))

        images_rgb = np.stack((images_r, images_g, images_b))
        images_rgb = np.transpose(images_rgb,axes=[1, 2, 3, 0])

        img_dir = os.path.join(self.root_folder, "images", self.split)
        os.makedirs(img_dir, exist_ok=True)
        print(f"Saving {self.split} images. This WILL take a while..")
        for i in tqdm(range(len(images_rgb)), ):
            img_ = images_rgb[i, :, :, :]
            im = Image.fromarray(img_)
            im.save(f"{img_dir}/{i}.jpg")
        
        save_array_to_txt(labels, os.path.join(self.root_folder, f"{self.split}.txt"))
        return

    def _load_dataset(self):
        def txt_to_list(file_path):
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                return [int(l.strip().rstrip()) for l in lines]
            except FileNotFoundError:
                print(f"Error: File not found at path: {file_path}")
                return []
            except Exception as e:
                print(f"An error occurred: {e}")
                return []
        cached_filename = os.path.join(self.root_folder,
                                       'cached_{:s}.pkl'.format(self.split))
        if os.path.exists(cached_filename):
            # load dataset into memory
            print("=> Loading from cached file {:s} ...".format(cached_filename))
            try:
                img_label_list = pickle.load(open(cached_filename, "rb"))
            except (RuntimeError, TypeError, NameError):
                print("Can't load cached file. Please remove the file and rebuild the cache!")
        else:
            # load dataset into memory
            print("Loading {:s} set into memory. This might take a while ...".format(self.split))
            img_label_list = tuple()
            labels_list = txt_to_list(os.path.join(self.root_folder, f"{self.split}.txt"))

            for i in tqdm(range(len(labels_list))):
                img = Image.open(os.path.join(self.root_folder, "images", self.split, f"{str(i)}.jpg")).convert('RGB')
                label = labels_list[i]
                img_label_list += ((img, label), )
            pickle.dump(img_label_list, open(cached_filename, "wb"))
        return img_label_list

    def __len__(self):
        return len(self.img_label_list)

    def __getitem__(self, index):
        # load img and label
        img, label = self.img_label_list[index]

        # apply data augmentation
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def get_index_mapping(self):
        # load the train label file
        train_label_file = os.path.join(self.root_folder, self.split + '.txt')
        if not os.path.exists(train_label_file):
            raise ValueError(
                'Label file {:s} does not exist!'.format(train_label_file))
        with open(train_label_file) as f:
            lines = f.readlines()

        # get the category names
        id_index_map = {}
        for line in lines:
            filename, label_id = line.rstrip('\n').split(' ')
            cat_name = filename.split('/')[-2]
            id_index_map[label_id] = cat_name

        # return a dictionary that maps an ID to its category name
        return id_index_map

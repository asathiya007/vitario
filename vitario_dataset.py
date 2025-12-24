import os
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_img_transforms(img_size):
    '''
    Input:
    img_size (tuple[int, int]) - tuple of image height and width

    Output:
    image transforms (torchvision.transforms.transforms.Compose) - image
    transforms that should be applied to images before they are passed to
    ViTario vision transformer (ViT) model
    '''
    return transforms.Compose([
        # resize image to expected size
        transforms.Resize(img_size),
        # convert PIL Image to torch Tensor, scales pixel values to range
        # [0.0, 1.0]
        transforms.ToTensor(),
        # scale pixel values to range [-1.0, 1.0]
        lambda i: i * 2.0 - 1.0
    ])


class ViTarioDataset(Dataset):
    '''
    Dataset of frames and actions, for training the ViTario vision transformer
    (ViT) model
    '''

    def __init__(self, dataset_dir, img_size):
        '''
        Input:
        dataset_dir (str) - path to dataset root directory

        Output:
        None
        '''

        # set dataset directory and get dataset DataFrame
        self.dataset_dir = dataset_dir
        self.dataset_df = self._get_dataset_df()

        # define image transforms
        self.img_size = img_size
        self.transforms = get_img_transforms(
            (self.img_size[1], self.img_size[2]))

    def _get_dataset_df(self):
        '''
        Input:
        None

        Output:
        dataset_df (pd.DataFrame) - DataFrame containing frame filepaths and
        corresponding actions
        '''

        # initialize dataset dictionary, for storing frame filepaths and
        # corresponding actions
        dataset_dict = {
            'frame': [],
            'action': []
        }

        # get frame directories corresponding to successful (win) playthroughs
        dataset_path = Path(self.dataset_dir)
        frame_dirs = [
            f.name for f in dataset_path.iterdir()
            if f.is_dir() and f.name[-4:] == '_win']

        # for each frame directory, get frame filenames and corresponding
        # actions, add to dataset
        for frame_dir in frame_dirs:
            frame_dir_path = Path(os.path.join(self.dataset_dir, frame_dir))
            frame_files = [f.name for f in frame_dir_path.glob('*.png')]
            frame_filepaths = list(map(
                lambda f: os.path.join(self.dataset_dir, frame_dir, f),
                frame_files))
            actions = list(map(lambda f: int(f.split('_')[5][1:]), frame_files))
            dataset_dict['frame'].extend(frame_filepaths)
            dataset_dict['action'].extend(actions)

        # create and return dataset DataFrame
        dataset_df = pd.DataFrame(dataset_dict)
        return dataset_df

    def __len__(self):
        '''
        Input:
        None

        Output:
        length of dataset (int) - number of (frame, action) pairs
        '''
        return len(self.dataset_df)

    def __getitem__(self, idx):
        '''
        Input:
        idx (int) - index of the (frame, action) pair to retrieve from the
        dataset

        Output:
        frame (torch.Tensor[float]) - tensor representing the retrieved image,
        shape is (channels, height, width)
        action (int) - integer representing the action corresponding to that
        frame
        '''

        # get row from dataset
        dataset_df_row = self.dataset_df.iloc[idx]

        # get frame, apply transforms
        frame_filepath = dataset_df_row['frame']
        frame = Image.open(frame_filepath).convert('RGB')
        frame = self.transforms(frame)

        # get action
        action = dataset_df_row['action']

        # return frame and action pair
        return frame, action

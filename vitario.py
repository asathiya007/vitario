import logging
import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from vitario_dataset import ViTarioDataset, get_img_transforms
from vitario_model import ViTarioModel


# set default save/load directory
DEFAULT_SAVE_DIR = './vitario_save_dir'

# dataset info
DATASET_DIR = './data-smb'
IMG_SIZE = (3, 240, 256)
NUM_MOVES = 256
BIT_IDX_TO_BUTTON_MAP = {
    0: 'A',
    1: 'up',
    2: 'left',
    3: 'B',
    4: 'start',
    5: 'right',
    6: 'down',
    7: 'select' 
}


def get_next_moves_from_ints(next_move_ints):
    '''
    Input:
    next_move_ints (list[int]) - a list of integers representing next moves

    Output:
    next_move_strs (list[str]) - a list of string representations of next moves
    '''

    # convert each next move integer into its string representation by
    # checking its bits
    next_move_strs = []
    for next_move in next_move_ints:
        next_move_buttons = []
        next_move_bin_str = format(next_move, '08b')
        for i in range(len(next_move_bin_str)):
            if next_move_bin_str[i] == '1':
                next_move_buttons.append(BIT_IDX_TO_BUTTON_MAP[i])
        next_move_str = ' + '.join(next_move_buttons)
        if len(next_move_str) == 0:
            next_move_str = '[no buttons pressed]'
        next_move_strs.append(next_move_str)
    return next_move_strs


class ViTario:
    '''
    A system that uses a custom vision transformer (ViT) language model to
    predict the best move to make at a given frame of the Super Mario Bros.
    game on the Nintendo Entertainment System (NES)
    '''
    def __init__(self):
        # get image transforms
        self.img_transforms = get_img_transforms((IMG_SIZE[1], IMG_SIZE[2]))

        # set device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # get logger
        self.logger = logging.getLogger('ViTario_Logger')
        self.logger.setLevel(logging.INFO)
        logging.basicConfig(stream=sys.stdout)        
    
    def _get_model_instance(self):
        '''
        Input:
        None

        Output:
        model (ViTarioModel) - an instance of the ViTarioModel class
        '''
        self.logger.info('Instantiating model...')

        # create an instance of the ViTarioModel class
        model = ViTarioModel(
            img_size=IMG_SIZE,
            patch_size=(16, 16),
            num_moves=NUM_MOVES,
            embed_dim=64,
            num_attn_heads=4,
            attn_head_size=32,
            hidden_size=128,
            num_hidden_layers=1,
            num_attn_blocks=6,
            device=self.device).to(self.device)
        self.logger.info('Instantiated model')

        # print number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f'Model has {num_params / 1e6}M parameters')

        # return ViTarioModel instance
        return model


    def train(self, batch_size, epochs=5, test_size=0.2):
        '''
        Input:
        batch_size (int) - size of each batch of inputs
        epochs (int) - number of epochs
        test_size (float) - percentage of data that should be in the test
        dataset
        '''
        self.logger.info('Preparing dataset and data loaders...')

        # get dataset
        dataset = ViTarioDataset(DATASET_DIR, IMG_SIZE)

        # split into train and test sets (using a generator for determinism)
        num_test_examples = int(test_size * len(dataset))
        num_train_examples = len(dataset) - num_test_examples
        train_dataset, test_dataset = random_split(
            dataset, [num_train_examples, num_test_examples],
            generator=torch.Generator().manual_seed(0))
        
        # create data loaders
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

        self.logger.info('Prepared dataset and data loaders')

        # instantiate model
        self.model = self._get_model_instance()
        self.model.train()

        # create optimizer
        optimizer = AdamW(self.model.parameters(), lr=5e-4)

        @torch.no_grad()
        def _evaluate_model():
            num_correct_test = 0
            for _, (frames, actions) in tqdm(
                    enumerate(test_dataloader), desc='Evaluation',
                    total=len(test_dataloader)):
                # move frames (shape is (batch size, channels, height, width)
                # and actions (shape is (batch size)) to device
                frames = frames.to(self.device)
                actions = actions.to(self.device)

                # get logits of classification token for each input in batch
                # shape is (batch size, number of moves)
                logits = self.model(frames)[:, 0]

                # update correctness of model
                preds = torch.argmax(logits, dim=1)
                num_correct_test += (actions == preds).type(
                    torch.int32).sum().item()
            # print test accuracy
            test_accuracy = num_correct_test / num_test_examples
            self.logger.info(f'Test accuracy: {test_accuracy}')

        # pretrain model
        self.logger.info('Training model...')
        for epoch in range(epochs):
            self.logger.info(f'Epoch {epoch + 1} of {epochs}')

            # training
            num_correct_train = 0
            for _, (frames, actions) in tqdm(
                    enumerate(train_dataloader), desc='Training',
                    total=len(train_dataloader)):
                # move frames (shape is (batch size, channels, height, width)
                # and actions (shape is (batch size)) to device
                frames = frames.to(self.device)
                actions = actions.to(self.device)

                # reset gradients
                optimizer.zero_grad()

                # get logits of classification token for each input in batch
                # shape is (batch size, number of moves)
                logits = self.model(frames)[:, 0]

                # calculate loss
                loss = F.cross_entropy(logits, actions, reduction='mean')

                # backpropagate loss to update weights
                loss.backward()
                optimizer.step()

                # update correctness of model
                preds = torch.argmax(logits, dim=1)
                num_correct_train += (actions == preds).type(
                    torch.int32).sum().item()
            # print train accuracy
            train_accuracy = num_correct_train / num_train_examples
            self.logger.info(f'Train accuracy: {train_accuracy}')

            # evaluate on test set
            _evaluate_model()
            
        self.model.eval()
        self.logger.info('Finished pretraining model')

    @torch.no_grad()
    def predict_next_move(self, frames):
        '''
        Input:
        frames (list[PIL.Image]) - list of frame images, for which to predict
        the next moves

        Output:
        next_moves (list[tuple(str, float)]) - list of next moves and their
        probabilities
        '''
        
        # transform each frame image
        transformed_imgs = []
        for frame in frames:
            transformed_imgs.append(self.img_transforms(frame))

        # stack to form a batch
        batch = torch.stack(transformed_imgs, dim=0).to(self.device)
        
        # pass through model, get classification token logits
        logits = self.model(batch)[:, 0]

        # apply softmax function to get move probabilities
        next_move_probs = F.softmax(logits, dim=1)

        # get the most likely moves and their probabilities
        preds = torch.max(next_move_probs, dim=1)
        next_moves = preds.indices.tolist()
        next_move_probs = preds.values.tolist()

        # format results into a string
        next_move_strs = get_next_moves_from_ints(next_moves)

        # return next move strings and probabilities
        return list(zip(next_move_strs, next_move_probs))

    def save(self, save_dir=DEFAULT_SAVE_DIR):
        # create save directory if it doesn't already exist
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # save model
        model_save_path = os.path.join(save_dir, 'model_state_dict.pth')
        torch.save(self.model.state_dict(), model_save_path)
        self.logger.info(
            f'Saved model parameters to {model_save_path}')

    def load(self, load_dir=DEFAULT_SAVE_DIR):
        # load model
        model_save_path = os.path.join(load_dir, 'model_state_dict.pth')
        self.model = self._get_model_instance()
        self.model.load_state_dict(torch.load(model_save_path))
        self.model.eval()
        self.logger.info(
            f'Loaded model parameters from {model_save_path}')

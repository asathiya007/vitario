import torch
import torch.nn as nn
import math


def get_flattened_img_patches(imgs, patch_height, patch_width):
    '''
    Input:
    img (torch.Tensor[float]) - images tensor, shape is (batch size, channels,
    height, width)
    patch_height (int) - height of patch
    patch_width (int) - width of patch

    Output:
    flattened_img_patches (torch.Tensor[float]) - tensor of flattened image
    patches, shape is (batch size, number of patches, channels * patch height *
    patch width)
    '''
    
    # check that patch height and width are valid
    img_height = imgs.shape[2]
    if img_height % patch_height != 0:
        raise Exception(
            f'Invalid patch height ({patch_height}). Must divide image height ' 
            + f'({img_height}) evenly.')
    img_width = imgs.shape[3]
    if img_width % patch_width != 0:
        raise Exception(
            f'Invalid patch width ({patch_width}). Must divide image width ' 
            + f'({img_width}) evenly.')

    # divide batch of images into patches (each patch has shape (batch size,
    # channels * patch height * patch width)) and flatten
    flattened_img_patches = []
    num_rows = img_height // patch_height
    num_cols = img_width // patch_width
    for i in range(num_rows):
        for j in range(num_cols):
            img_patch = imgs[:, :, i * patch_height: (i + 1) * patch_height,
                           j * patch_width: (j + 1) * patch_width]
            flattened_img_patches.append(
                img_patch.flatten(start_dim=1, end_dim=-1))
    
    # return flattened image patches as a single tensor, with shape (batch size,
    # number of patches, channels * patch height * patch width)
    flattened_img_patches = torch.stack(flattened_img_patches, dim=1)
    return flattened_img_patches


class SelfAttnHead(nn.Module):
    def __init__(self, attn_head_size, context_size):
        '''
        Input:
        attn_head_size (int) - number of dimensions of projection layer outputs
        as well as attention head output
        context_size (int) - number of tokens in a single input
        '''
        super().__init__()
        self.attn_head_size = attn_head_size
        self.context_size = context_size

        # query, key, and value projection layers
        self.q_proj = nn.Linear(self.attn_head_size, self.attn_head_size)
        self.k_proj = nn.Linear(self.attn_head_size, self.attn_head_size)
        self.v_proj = nn.Linear(self.attn_head_size, self.attn_head_size)

    def _get_init_attn_pattern(self, q, k):
        '''
        Input:
        q (torch.Tensor[float]) - tensor containing query vectors, shape is
        (batch size, context size, attn head size)
        k (torch.Tensor[float]) - tensor containing key vectors, shape is
        (batch size, context size, attn head size)

        Output:
        initial attention pattern (torch.Tensor[float]) - initial attention
        pattern, shape is (batch size, context size, context size)
        '''
        return q @ k.mT
    
    def _scale_attn_pattern(self, attn_pattern, scaling_factor):
        '''
        Input:
        attn_pattern (torch.Tensor[float]) - attention pattern, shape is
        (batch size, context size, context size)

        Output:
        scaled attention pattern (torch.Tensor[float]) - scaled attention pattern,
        shape is (batch size, context size, context size)
        '''
        return attn_pattern * scaling_factor
    
    def _normalize_attn_pattern(self, attn_pattern):
        '''
        Input:
        attn_pattern (torch.Tensor[float]) - attention pattern, shape is
        (batch size, context size, context size)

        Output:
        normalized attention pattern (torch.Tensor[float]), where all rows sum
        to 1
        '''
        return torch.softmax(attn_pattern, dim=-1)

    def forward(self, mha_q, mha_k, mha_v):
        '''
        Input:
        mha_q (torch.Tensor[float]) - query vectors, shape is
        (batch size, context size, attn head size)
        mha_k (torch.Tensor[float]) - key vectors, shape is
        (batch size, context size, attn head size)
        mha_v (torch.Tensor[float]) - value vectors, shape is
        (batch size, context size, attn head size)

        Output:
        self-attention head output (torch.Tensor[float]) - information collected
        across token embeddings to enrich them, shape is
        (batch size, context size, attn head size)
        '''

        # get head-specific query, key, and value projections
        # input: mha_q, mha_k, mha_v (initial query, key, and value vectors,
        # given to all attention heads)
        # input shapes: (batch size, context size, attn head size),
        # (batch size, context size, attn head size),
        # (batch size, context size, attn head size)
        # outputs: q, k, v
        # output shapes: (batch size, context size, attn head size),
        # (batch size, context size, attn head size),
        # (batch size, context size, attn head size)
        q = self.q_proj(mha_q)
        k = self.k_proj(mha_k)
        v = self.v_proj(mha_v)

        # get attention pattern
        init_attn_pattern = self._get_init_attn_pattern(q, k)
        # scaled attention pattern (scaled for numerical stability)
        scaled_attn_pattern = self._scale_attn_pattern(
            init_attn_pattern,
            1 / torch.sqrt(torch.Tensor([self.attn_head_size])).item())
        # normalize attention pattern with softmax function, across rows
        normalized_attn_pattern = self._normalize_attn_pattern(
            scaled_attn_pattern)

        # collect info across multiple token embeddings
        # inputs: normalized attn pattern, v
        # input shapes: (batch size, context size, context size),
        # (batch size, context size, attn head size)
        # output: information collected across
        # output shape: (batch size, context size, attn head size)
        collected_info = normalized_attn_pattern @ v
        return collected_info


class MultiHeadAttn(nn.Module):
    def __init__(self, embed_dim, context_size, num_attn_heads, attn_head_size):
        '''
        Input:
        embed_dim (int) - number of dimensions of token embedding
        context_size (int) - number of tokens in a single input
        num_attn_heads (int) - number of self-attention heads
        attn_head_size (int) - number of dimensions of projection layer outputs
        as well as attention head output
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.num_attn_heads = num_attn_heads
        self.attn_head_size = attn_head_size

        # query, key, and value projection layers
        self.mha_q_proj = nn.Linear(self.embed_dim, self.attn_head_size)
        self.mha_k_proj = nn.Linear(self.embed_dim, self.attn_head_size)
        self.mha_v_proj = nn.Linear(self.embed_dim, self.attn_head_size)
    
        # create attention heads
        self.attn_heads = nn.ModuleList([])
        for _ in range(self.num_attn_heads):
            self.attn_heads.append(SelfAttnHead(
                self.attn_head_size, self.context_size))

        # output projection layer
        self.o_proj = nn.Linear(
            self.attn_head_size * self.num_attn_heads, self.embed_dim)

    def _get_concat_sah_outputs(self, x):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of token embeddings, shape is
        (batch size, context size, embed dim)

        Output:
        concatenated self-attention head outputs (torch.Tensor[float]) - tensor
        of concatentated self-attention head outputs, shape is 
        (batch size, context size, attn head size * num attn heads)
        '''

        # get query, key, and value projections of token embeddings
        # input: x (token embeddings)
        # input shapes: (batch size, context size, embed dim),
        # (batch size, context size, embed dim),
        # (batch size, context size, embed dim)
        # outputs: q, k, v
        # output shapes: (batch size, context size, attn head size),
        # (batch size, context size, attn head size),
        # (batch size, context size, attn head size)
        mha_q = self.mha_q_proj(x)
        mha_k = self.mha_k_proj(x)
        mha_v = self.mha_v_proj(x)

        # get concatenated attention head outputs
        self_attn_head_outputs = [
            sah(mha_q, mha_k, mha_v) for sah in self.attn_heads]
        concat_sah_outputs = torch.cat(self_attn_head_outputs, dim=-1)
        return concat_sah_outputs

    def forward(self, x):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of token embeddings, shape is
        (batch size, context size, embed dim)

        Output:
        multi-head attention output - information collected across token
        embeddings to enrich them, shape is
        (batch size, context size, embed dim)
        '''

        # get information to enrich token embeddings from each self-attention
        # head
        concat_sah_outputs = self._get_concat_sah_outputs(x)

        # project the concatenated self-attention head outputs to the embedding
        # space to get the multi-head attention output (information to enrich
        # the token embeddings)
        # input: cocnatenated self-attention head outputs
        # input shape: (batch size, context size, attn head size * num attn
        # heads)
        # output: multi-head attention output
        # output shape: (batch size, context size, embed dim)
        multi_head_attn_output = self.o_proj(concat_sah_outputs) 

        # return multi-head attention output
        return multi_head_attn_output


class FeedFwd(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_hidden_layers):
        '''
        Input:
        embed_dim (int) - number of dimensions of token embedding
        hidden_size (int) - hidden size of the feed-forward network
        num_hidden_layers (int) - number of hidden layers in the feed-forward
        network
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # define layers of network
        input_layer = [
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU()
        ]
        hidden_layers = []
        for _ in range(self.num_hidden_layers):
            hidden_layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])
        output_layer = [
            nn.Linear(hidden_size, embed_dim)
        ]
        self.layers = nn.Sequential(
            *(input_layer + hidden_layers + output_layer))
    
    def forward(self, x):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of enriched token embeddings, shape is
        (batch_size, context size, embed dim)

        Output:
        feed-forward network output (torch.Tensor[float]) - tensor containing
        information to further enrich token embeddings, shape is
        (batch size, context size, embed dim)
        '''
        return self.layers(x)


class AttnBlock(nn.Module):
    def __init__(self, embed_dim, context_size, num_attn_heads, attn_head_size,
                 hidden_size, num_hidden_layers):
        '''
        Input:
        embed_dim (int) - number of dimensions of token embedding
        context_size (int) - number of tokens in a single input
        num_attn_heads (int) - number of self-attention heads
        attn_head_size (int) - number of dimensions of attention projection
        layer outputs as well as attention head output
        hidden_size (int) - hidden size of the feed-forward network
        num_hidden_layers (int) - number of hidden layers in the feed-forward
        network
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.context_size = context_size
        self.num_attn_heads = num_attn_heads
        self.attn_head_size = attn_head_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # multi-head attention
        self.mah = MultiHeadAttn(
            self.embed_dim, self.context_size, self.num_attn_heads,
            self.attn_head_size)

        # layer normalization
        self.post_mah_layer_norm = nn.LayerNorm(self.embed_dim)
        self.post_ffwd_layer_norm = nn.LayerNorm(self.embed_dim)

        # feed forward network
        self.ffwd = FeedFwd(
            self.embed_dim, self.hidden_size, self.num_hidden_layers)

    def forward(self, x):
        '''
        Input:
        x (torch.Tensor[float]) - tensor of token embeddings, shape is
        (batch size, context size, embed dim)

        Output:
        attention block output (torch.Tensor[float]) - tensor of enriched token
        embeddings, shape is (batch size, context size, embed dim)
        '''
        # enrich token embeddings with multi-head attention
        x = x + self.mah(x)

        # layer normalization
        x = self.post_mah_layer_norm(x)

        # enrich token embeddings with feed forward network
        x = x + self.ffwd(x)

        # layer normalization
        x = self.post_ffwd_layer_norm(x)

        # return enriched token embeddings
        return x


class ViTarioModel(nn.Module):
    '''
    A vision transformer (ViT) model for predicting the next move given a frame
    in the Super Mario Bros. game for the Nintendo Entertainment System (NES)
    '''
    def __init__(self, img_size, patch_size, num_moves, embed_dim,
                 num_attn_heads, attn_head_size, hidden_size, num_hidden_layers,
                 num_attn_blocks, device):
        '''
        Input:
        img_size (tuple(int, int, int)) - size of each image 
        patch_size (tuple(int, int)) - height and width of each image patch
        num_moves (int) - total number of moves
        embed_dim (int) - number of dimensions of token embedding
        num_attn_heads (int) - number of self-attention heads per attention
        block
        attn_head_size (int) - number of dimensions of attention projection
        layer outputs as well as attention head output
        hidden_size (int) - hidden size of the feed-forward network in each
        attention block
        num_hidden_layers (int) - number of hidden layers in the feed-forward
        network of each attention block
        num_attn_blocks (int) - number of attention blocks
        device (torch.device) - the device that the model is running on
        '''
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # check validity of image size and patch size
        if self.img_size[1] % self.patch_size[0] != 0:
            raise Exception(
                f'Image height {self.img_size[0]} is not evenly divisble by '
                + f'patch height {self.patch_size[0]}')
        if self.img_size[2] % self.patch_size[1] != 0:
            raise Exception(
                f'Image width {self.img_size[1]} is not evenly divisble by '
                + f'patch width {self.patch_size[1]}')
        # context size is equal to the number of patches + 1 (classification
        # token)
        self.context_size = (self.img_size[0] // self.patch_size[0]) * (
            self.img_size[1] // self.patch_size[1]) + 1
        self.num_moves = num_moves
        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads
        self.attn_head_size = attn_head_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attn_blocks = num_attn_blocks
        self.device = device

        # image patches embedding
        self.img_patch_embed = nn.Linear(
            self.img_size[0] * self.patch_size[0] * self.patch_size[1],
            self.embed_dim)
        
        # classification token embedding
        self.cls_token_embed = nn.Embedding(1, self.embed_dim)

        # sinusoidal positional embedding
        self.sin_pos_embeds = self._get_sin_pos_embeds().to(self.device)

        # attention blocks
        self.attn_blocks = nn.ModuleList([])
        for _ in range(self.num_attn_blocks):
            self.attn_blocks.append(AttnBlock(
                self.embed_dim, self.context_size, self.num_attn_heads,
                self.attn_head_size, self.hidden_size, self.num_hidden_layers))

        # output projection layer
        self.o_proj = nn.Linear(self.embed_dim, self.num_moves)

    def _get_sin_pos_embeds(self):
        '''
        Computes sinusoidal positional embeddings.
        
        Input:
        None

        Output:
        sinusoidal positional embeddings (torch.Tensor[float]) - tensor of
        sinusoidal positional embeddings, shape is (context size,
        embed_dim)
        '''

        # get positions vector, shape is (context size, )
        positions = torch.arange(self.context_size)

        # compute frequencies (log is used to avoid overflow errors, when
        # raising 10000 to a large power) across all positions, shape is
        # (context size, half embed dim)
        half_embed_dim = math.ceil(self.embed_dim / 2)
        frequencies = positions[:, None] * torch.exp(
            -1 * (2 / self.embed_dim) * torch.arange(half_embed_dim)
            * math.log(10000))[None, :]

        # get sin and cos values, interleave them (by stacking along a new
        # dimension and reshaping), and truncate at original embed dim
        # to get positional embeddings of shape (context size, embed dim)
        sin_values = frequencies.sin()
        cos_values = frequencies.cos()
        sin_pos_embeds = torch.stack([sin_values, cos_values], dim=-1)\
            .reshape(self.context_size, half_embed_dim * 2)
        sin_pos_embeds = sin_pos_embeds[:, :self.embed_dim]

        # return sinusoidal positional embeddings
        return sin_pos_embeds

    def forward(self, x):
        '''
        Input:
        x (torch.Tensor[float]) - image tensor, shape is (batch size, channels,
        image height, image width)

        Output:
        transformer model output (torch.Tensor[float]) - tensor of logits
        (enriched token embeddings, projected from embed_dim dimensions
        to num_moves dimensions), shape is (batch size, context size,
        number of moves)
        '''

        # get flattened image patches ("image tokens"), shape is
        # (batch size, context size - 1, channels * patch height * patch width)
        flattened_img_patches = get_flattened_img_patches(
            x, self.patch_size[0], self.patch_size[1])
        
        # get image token embeddings (via linear projection) and concatenate
        # with classification token embedding to get input token embeddings,
        # shape is (batch size, context size, embed dim)
        img_patch_embeds = self.img_patch_embed(flattened_img_patches)
        cls_token_embed = self.cls_token_embed(
            torch.zeros((flattened_img_patches.shape[0], 1))
                .type(torch.int32).to(self.device))
        input_tok_embeds = torch.cat([cls_token_embed, img_patch_embeds], dim=1)

        # enrich token embeddings with positional info
        enriched_tok_embeds = input_tok_embeds + self.sin_pos_embeds

        # enrich token embeddings through attention blocks
        for attn_block in self.attn_blocks:
            enriched_tok_embeds = attn_block(enriched_tok_embeds)

        # project token embeddings from embed_dim dimensions to num_moves
        # dimensions to get logits
        logits = self.o_proj(enriched_tok_embeds)

        # return logits
        return logits

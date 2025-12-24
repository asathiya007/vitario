# ViTario

ViTario is an AI system that uses a custom vision transformer (ViT) model to predict the next move to make at a given frame in the Super Mario Bros. game on the Nintendo Entertainment System (NES).

See the `vitario.ipynb` notebook for a demo.

The ViTario ViT model is trained on data from this dataset:
- smbdataset by Rafael C.P.: [https://github.com/rafaelcp/smbdataset](https://github.com/rafaelcp/smbdataset). This dataset is licensed under the Attribution 4.0 International license, more info is available here: [https://creativecommons.org/licenses/by/4.0/legalcode.en](https://creativecommons.org/licenses/by/4.0/legalcode.en). The data is preprocessed (see code for more details) before it is used to train the ViT model.

ViTario's implementation is based on the following sources.
- "Let's build GPT: from scratch, in code, spelled out" video by Andrej Karpathy: [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Google Colab notebook for "Let's build GPT: from scratch, in code, spelled out" video by Andrej Karpathy: [https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing)
- "Attention is All You Need" paper by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin: [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)
- "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" paper by 
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby: [https://arxiv.org/pdf/2010.11929](https://arxiv.org/pdf/2010.11929)

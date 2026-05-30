import torch
import torch.nn as nn
from models.encoder import CNNEncoder
from models.decoder import TransformerDecoder

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size, decoder_dim, attention_dim, dropout=0.1, max_len=50):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            decoder_dim=decoder_dim,
            dropout=dropout,
            max_len=max_len
        )
        self.max_len = max_len

    def generate_square_subsequent_mask(self, sz):
        """Causal mask for transformer decoder"""
        mask = torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)
        return mask

    def forward(self, images, captions):
        """
        images: (B, 3, H, W)
        captions: (B, seq_len)
        """
        memory = self.encoder(images)  # (B, 196, embed_size)

        tgt_mask = self.generate_square_subsequent_mask(captions.size(1)).to(images.device)

        outputs = self.decoder(
            tgt=captions,
            memory=memory,
            tgt_mask=tgt_mask
        )  # (B, seq_len, vocab_size)

        return outputs

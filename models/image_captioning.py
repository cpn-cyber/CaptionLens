import torch
import torch.nn as nn
from models.encoder import CNNEncoder
from models.decoder import TransformerDecoder
from models.semantic_slot import SemanticSlotMemoryAdapter

class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        decoder_dim,
        attention_dim,
        dropout=0.1,
        max_len=50,
        use_semantic_slots=False,
        num_semantic_slots=16,
        semantic_slot_layers=2,
        semantic_slot_heads=8,
        include_visual_tokens=True,
    ):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.semantic_slot_adapter = None
        if use_semantic_slots:
            self.semantic_slot_adapter = SemanticSlotMemoryAdapter(
                embed_size=embed_size,
                num_slots=num_semantic_slots,
                num_heads=semantic_slot_heads,
                num_layers=semantic_slot_layers,
                dropout=dropout,
                include_visual_tokens=include_visual_tokens,
            )
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
        memory = self.encode_image(images)

        tgt_mask = self.generate_square_subsequent_mask(captions.size(1)).to(images.device)

        outputs = self.decoder(
            tgt=captions,
            memory=memory,
            tgt_mask=tgt_mask
        )  # (B, seq_len, vocab_size)

        return outputs

    def encode_image(self, images):
        memory = self.encoder(images)  # (B, 196, embed_size)
        if self.semantic_slot_adapter is not None:
            memory = self.semantic_slot_adapter(memory)
        return memory

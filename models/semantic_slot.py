import torch
import torch.nn as nn


class SemanticSlotBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_size)
        self.self_norm = nn.LayerNorm(embed_size)
        self.ffn_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * 4, embed_size),
        )

    def forward(self, slots, visual_tokens):
        attended_slots, _ = self.cross_attention(
            query=slots,
            key=visual_tokens,
            value=visual_tokens,
            need_weights=False,
        )
        slots = self.cross_norm(slots + self.dropout(attended_slots))

        contextual_slots, _ = self.self_attention(
            query=slots,
            key=slots,
            value=slots,
            need_weights=False,
        )
        slots = self.self_norm(slots + self.dropout(contextual_slots))
        slots = self.ffn_norm(slots + self.dropout(self.ffn(slots)))
        return slots


class SemanticSlotMemoryAdapter(nn.Module):
    def __init__(
        self,
        embed_size,
        num_slots=16,
        num_heads=8,
        num_layers=2,
        num_visual_tokens=196,
        dropout=0.1,
        include_visual_tokens=True,
    ):
        super().__init__()
        self.include_visual_tokens = include_visual_tokens
        self.slot_queries = nn.Parameter(torch.randn(num_slots, embed_size) * 0.02)
        self.visual_pos_embedding = nn.Parameter(
            torch.randn(1, num_visual_tokens, embed_size) * 0.02
        )
        self.blocks = nn.ModuleList([
            SemanticSlotBlock(embed_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(embed_size)

    def forward(self, visual_tokens):
        batch_size, token_count, _ = visual_tokens.shape
        if token_count <= self.visual_pos_embedding.size(1):
            visual_tokens = visual_tokens + self.visual_pos_embedding[:, :token_count]

        slots = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            slots = block(slots, visual_tokens)
        slots = self.output_norm(slots)

        if self.include_visual_tokens:
            return torch.cat([slots, visual_tokens], dim=1)
        return slots

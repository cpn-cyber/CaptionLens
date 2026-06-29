# infer.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import yaml

from models.image_captioning import ImageCaptioningModel
from utils.decoding import generate_caption as decode_image
from utils.tokenizer import Tokenizer

# Load config
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = Tokenizer()
tokenizer.load_vocab(config['processed_data']['vocab_path'])

# Load model
model = ImageCaptioningModel(
    vocab_size=len(tokenizer.word2idx),
    embed_size=config['model']['embed_size'],
    decoder_dim=config['model']['decoder_dim'],
    attention_dim=config['model']['attention_dim'],
    dropout=config['model']['dropout'],
    max_len=config['model']['max_len'],
    use_semantic_slots=config['model'].get('use_semantic_slots', False),
    num_semantic_slots=config['model'].get('num_semantic_slots', 16),
    semantic_slot_layers=config['model'].get('semantic_slot_layers', 2),
    semantic_slot_heads=config['model'].get('semantic_slot_heads', 8),
    include_visual_tokens=config['model'].get('include_visual_tokens', True),
)
model.load_state_dict(torch.load(config['train']['save_dir'] + "best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Caption generation
def generate_caption(image_path, max_len=None, strategy="beam"):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    decoding_config = dict(config["decoding"])
    if max_len is not None:
        decoding_config["max_len"] = max_len
    return decode_image(
        model,
        image_tensor,
        tokenizer,
        strategy=strategy,
        **decoding_config,
    )

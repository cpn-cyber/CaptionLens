# infer.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import yaml

from models.image_captioning import ImageCaptioningModel
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
    max_len=config['model']['max_len']
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

# Caption generation (Greedy decoding)
def generate_caption(image_path, max_len=50):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        memory = model.encoder(image)  # (1, 196, E)

        caption_idxs = [tokenizer.word2idx["<START>"]]

        for _ in range(max_len):
            caption_tensor = torch.tensor(caption_idxs, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)
            tgt_mask = model.generate_square_subsequent_mask(caption_tensor.size(1)).to(device)

            output = model.decoder(caption_tensor, memory, tgt_mask=tgt_mask)
            next_token = output[:, -1, :].argmax(dim=-1).item()

            caption_idxs.append(next_token)

            if next_token == tokenizer.word2idx["<END>"]:
                break

    caption = tokenizer.decode(caption_idxs[1:-1])  # Remove <START> and <END>
    return caption
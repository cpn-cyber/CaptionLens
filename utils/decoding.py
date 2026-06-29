import torch


def _blocked_tokens(sequence, ngram_size):
    if ngram_size <= 0 or len(sequence) + 1 < ngram_size:
        return set()

    if ngram_size == 1:
        return set(sequence)

    prefix = tuple(sequence[-(ngram_size - 1):])
    blocked = set()
    for index in range(len(sequence) - ngram_size + 1):
        if tuple(sequence[index:index + ngram_size - 1]) == prefix:
            blocked.add(sequence[index + ngram_size - 1])
    return blocked


def _next_log_probs(model, memory, token_ids):
    device = memory.device
    caption_tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    tgt_mask = model.generate_square_subsequent_mask(
        caption_tensor.size(1)
    ).to(device)
    output = model.decoder(caption_tensor, memory, tgt_mask=tgt_mask)
    return torch.log_softmax(output[:, -1, :], dim=-1).squeeze(0)


def _caption_from_tokens(token_ids, tokenizer, drop_unk=True):
    excluded = {
        tokenizer.word2idx["<PAD>"],
        tokenizer.word2idx["<START>"],
        tokenizer.word2idx["<END>"],
    }
    if drop_unk:
        excluded.add(tokenizer.word2idx["<UNK>"])

    filtered = [token_id for token_id in token_ids if token_id not in excluded]
    return tokenizer.decode(filtered).strip()


def greedy_decode(model, memory, tokenizer, max_len=24, drop_unk=True):
    start_idx = tokenizer.word2idx["<START>"]
    end_idx = tokenizer.word2idx["<END>"]
    token_ids = [start_idx]

    for _ in range(max_len):
        log_probs = _next_log_probs(model, memory, token_ids)
        next_token = int(torch.argmax(log_probs).item())
        token_ids.append(next_token)
        if next_token == end_idx:
            break

    return _caption_from_tokens(token_ids, tokenizer, drop_unk=drop_unk)


def beam_search_decode(
    model,
    memory,
    tokenizer,
    max_len=24,
    beam_size=5,
    length_penalty=0.7,
    repetition_penalty=0.8,
    unk_penalty=4.0,
    no_repeat_ngram_size=2,
    drop_unk=True,
):
    start_idx = tokenizer.word2idx["<START>"]
    end_idx = tokenizer.word2idx["<END>"]
    unk_idx = tokenizer.word2idx["<UNK>"]
    beams = [([start_idx], 0.0, False)]

    def normalized_score(candidate):
        token_ids, score, _ = candidate
        return score / (len(token_ids) ** length_penalty)

    for _ in range(max_len):
        candidates = []

        for token_ids, score, ended in beams:
            if ended:
                candidates.append((token_ids, score, ended))
                continue

            log_probs = _next_log_probs(model, memory, token_ids)
            log_probs[unk_idx] -= unk_penalty

            for token_idx in set(token_ids[1:]):
                log_probs[token_idx] -= repetition_penalty

            for token_idx in _blocked_tokens(
                token_ids[1:],
                no_repeat_ngram_size,
            ):
                log_probs[token_idx] = -float("inf")

            top_scores, top_indices = torch.topk(log_probs, beam_size)
            for token_score, token_idx in zip(
                top_scores.tolist(),
                top_indices.tolist(),
            ):
                next_tokens = token_ids + [token_idx]
                candidates.append((
                    next_tokens,
                    score + token_score,
                    token_idx == end_idx,
                ))

        beams = sorted(
            candidates,
            key=normalized_score,
            reverse=True,
        )[:beam_size]

        if all(ended for _, _, ended in beams):
            break

    best_tokens = max(beams, key=normalized_score)[0]
    return _caption_from_tokens(best_tokens, tokenizer, drop_unk=drop_unk)


def decode_from_memory(model, memory, tokenizer, strategy="beam", **kwargs):
    if strategy == "greedy":
        allowed = {"max_len", "drop_unk"}
        greedy_kwargs = {
            key: value for key, value in kwargs.items() if key in allowed
        }
        return greedy_decode(
            model,
            memory,
            tokenizer,
            **greedy_kwargs,
        )

    if strategy == "beam":
        return beam_search_decode(
            model,
            memory,
            tokenizer,
            **kwargs,
        )

    raise ValueError(f"Unsupported decoding strategy: {strategy}")


def generate_caption(model, image_tensor, tokenizer, strategy="beam", **kwargs):
    with torch.no_grad():
        if hasattr(model, "encode_image"):
            memory = model.encode_image(image_tensor)
        else:
            memory = model.encoder(image_tensor)
        return decode_from_memory(
            model,
            memory,
            tokenizer,
            strategy=strategy,
            **kwargs,
        )

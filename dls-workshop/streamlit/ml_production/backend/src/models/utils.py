import torch
def mask_tokens(inputs,tokenizer,prob = 0.15):
    inputs = inputs['input_ids']

    labels = inputs.clone()

    # Create a probability matrix for masking
    probability_matrix = torch.full(labels.shape, prob)

    # Ensure [CLS], [SEP], and [PAD] tokens are not masked
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Create masked indices
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Set labels for non-masked tokens to -100 (ignored in loss)

    # Replace 80% of masked tokens with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # Replace 10% of masked tokens with random words
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]


def load_model(model,cp):
    for name,param in model.named_parameters():
        if param.requires_grad:
            try:
                param.requires_grad_(False)
                p = torch.load(f"{cp}/{name}",map_location=model.device).requires_grad_(False)
                param.copy_(p)
                param.requires_grad_(True)
                param.retain_grad()
            except Exception as e:
                print(name, ' ', e)
                param.requires_grad_(True)
                param.retain_grad()

import torch
import numpy as np

def load_model(model,cp= "my_awesome_model_/checkpoint-3200"):
    for name,param in model.named_parameters():
        if param.requires_grad:
            try:
                param.requires_grad_(False)
                p = torch.load(f"{cp}/{name}").requires_grad_(False)
                param.copy_(p)
                param.requires_grad_(True)
                param.retain_grad()
            except Exception as e:
                print(name, ' ', e)
                param.requires_grad_(True)
                param.retain_grad()


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





def generate_related_contrastive_pairs(cls_token, desc_token, out_classes, labels,class_weights):
    


    """
    Generate all possible positive and negative pairs for contrastive learning.
    
    Args:
    cls_token: Tensor of shape (batch_size, embedding_dim)
    desc_token: Tensor of shape (k_descriptions*batch_size, embedding_dim)
    out_classes: Tensor of shape (k_descriptions*batch_size,), indices for each description in the label matrix.
    labels: Label matrix of shape (batch_size, num_classes)
    """
    batch_size, embedding_dim = cls_token.size()
    
    desc_token_batched = desc_token.view(batch_size,-1,embedding_dim)
    num_descriptions = desc_token_batched.size(1)


    cls_token = cls_token.unsqueeze(1).repeat(1, num_descriptions, 1).view(-1, embedding_dim)  # Shape: (batch_size * num_descriptions, embedding_dim)

   
    out_classes = out_classes.view(batch_size,num_descriptions)
    pair_labels = []
    weights = []
    for i in range(batch_size):
        w = np.array(class_weights)*labels[i].int().detach().cpu().numpy() + np.array(1-class_weights)*(1-labels[i]).int().detach().cpu().numpy()
        for j in range(num_descriptions):
            pair_labels.append(labels[i].int()[out_classes[i,j].int()])
            weights.append(w[[out_classes[i,j].int().detach().cpu()]])

    pair_labels = torch.Tensor(pair_labels).cuda().view(-1)


    return cls_token, desc_token, pair_labels,weights




def compute_weights(train_dataset, temperature = 0.16,gamma = 2):
    import numpy as np
    import torch
    temperature = temperature

    pos_count = torch.Tensor(train_dataset['labels']).sum(dim=0)
    neg_count = torch.Tensor(train_dataset['labels']).shape[0]-pos_count

    pos_count = 1/torch.sqrt(pos_count * temperature+1)
    neg_count = 1/torch.sqrt(neg_count * temperature+1)

    weights = ((neg_count)/(pos_count)*gamma).cuda()
    return weights.detach().cpu().numpy()
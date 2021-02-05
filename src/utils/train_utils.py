import torch 

def label_noise(labels,frac):
    '''
        Invert a certain fraction of the labels

        Params:
            labels: Torch tensor.
            frac: proportion of labels to invert.
    '''
    tensor_size = len(labels)
    # Fraction of labels to change
    n_labels = int(frac * tensor_size)
    # Select labels to change
    mask = torch.randperm(tensor_size)[n_labels]
    # Flip labels
    labels[mask] = 0.9 - labels[mask]
    return labels
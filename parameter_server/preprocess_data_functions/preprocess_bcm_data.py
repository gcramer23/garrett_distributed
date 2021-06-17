def preprocess_bcm_data(rank, data):
    r"""
    A method to move the Bert data from CPU to GPU. For each sample it moves
    all tensors in the dictionary to the GPU.
    Args:
        rank (int): worker rank
        data (list): Bert training samples
    """

    def process_sample(sample):
        p_sample = {}
        for k, v in sample.items():
            if type(v) is dict:
                p_sample[k] = process_sample(v)
            else:
                v = v.cuda(rank)
                p_sample[k] = v
        return p_sample

    for i, sample in enumerate(data):
        data[i] = process_sample(sample)
    return data

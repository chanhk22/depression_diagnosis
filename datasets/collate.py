import torch

def pad_to_max(batch_list, pad_value=0.0):
    # 윈도우 길이는 동일(고정)이라 패딩 불필요. 바로 stack.
    return torch.stack(batch_list, 0)

def collate_fn(samples):
    # samples: list of dicts (from WindowDataset)
    out = {}
    # keep list fields
    out['pid'] = [s['pid'] for s in samples]
    out['domain'] = [s['domain'] for s in samples]
    out['split'] = [s['split'] for s in samples]
    out['file'] = [s['file'] for s in samples]

    # numeric arrays: lld, lmk, micro, priv (may be missing)
    if 'lld' in samples[0]:
        out['lld'] = pad_stack([torch.tensor(s['lld'], dtype=torch.float32) for s in samples])
    if 'lmk' in samples[0]:
        out['lmk'] = pad_stack([torch.tensor(s['lmk'], dtype=torch.float32) for s in samples if 'lmk' in s])
    if 'micro' in samples[0]:
        out['micro'] = pad_stack([torch.tensor(s['micro'], dtype=torch.float32) for s in samples if 'micro' in s])
    if 'priv' in samples[0]:
        out['priv'] = pad_stack([torch.tensor(s['priv'], dtype=torch.float32) for s in samples if 'priv' in s])

    out['label'] = torch.tensor([int(s['label']) for s in samples], dtype=torch.float32)
    return out

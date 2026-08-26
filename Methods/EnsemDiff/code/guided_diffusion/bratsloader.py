import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
from pathlib import Path


def _seqtype_from_name(filename):
    parts = filename.split("_")
    if len(parts) < 4:
        return None
    token = parts[3]
    for suffix in (".nii.gz", ".nii", ".npy"):
        if token.endswith(suffix):
            token = token[: -len(suffix)]
    return token


def _load_image(path):
    if path.endswith(".npy"):
        return np.load(path)
    return nibabel.load(path).get_fdata()


def _crop_to_224(tensor):
    h, w = tensor.shape[-2:]
    if (h, w) == (224, 224):
        return tensor
    if h < 224 or w < 224:
        raise ValueError(f"Expected at least 224x224 spatial size, got {(h, w)}")
    top = (h - 224) // 2
    left = (w - 224) // 2
    return tensor[..., top : top + 224, left : left + 224]


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    if f.startswith("._"):
                        continue
                    if not (f.endswith(".nii.gz") or f.endswith(".nii") or f.endswith(".npy")):
                        continue
                    seqtype = _seqtype_from_name(f)
                    if seqtype not in {'t1', 't1ce', 't2', 'flair', 'seg'}:
                        continue
                    datapoint[seqtype] = os.path.join(root, f)
                assert self.seqtypes_set.issubset(datapoint.keys()), \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append({k: datapoint[k] for k in self.seqtypes})

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            path=filedict[seqtype]
            out.append(torch.as_tensor(_load_image(path), dtype=torch.float32))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = _crop_to_224(image)     # crop to the paper's (224, 224) size
            return (image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = _crop_to_224(image)      # crop to the paper's (224, 224) size
            label = _crop_to_224(label)
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            return (image, label)

    def __len__(self):
        return len(self.database)


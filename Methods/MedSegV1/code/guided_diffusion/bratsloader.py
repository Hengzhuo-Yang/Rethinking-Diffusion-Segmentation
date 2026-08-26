import os
import os.path
from collections import OrderedDict
import json

import nibabel
import numpy as np
import torch
import torch.nn
import torchvision.utils as vutils


SEQTYPE_ALIASES = {
    "t1": "t1",
    "t1ce": "t1ce",
    "t1c": "t1ce",
    "t1gd": "t1ce",
    "t2": "t2",
    "flair": "flair",
    "t2flair": "flair",
    "seg": "seg",
}


def _strip_nii_suffix(filename):
    lower = filename.lower()
    for suffix in (".nii.gz", ".nii"):
        if lower.endswith(suffix):
            return lower[: -len(suffix)]
    return lower


def _seqtype_from_filename(filename):
    stem = _strip_nii_suffix(filename)
    parts = stem.split("_")
    for part in reversed(parts):
        if part in SEQTYPE_ALIASES:
            return SEQTYPE_ALIASES[part]
    if len(parts) > 3 and parts[3] in SEQTYPE_ALIASES:
        return SEQTYPE_ALIASES[parts[3]]
    raise ValueError(f"cannot infer BraTS sequence type from {filename}")


def _iter_nifti_files(files):
    for filename in files:
        lower = filename.lower()
        if lower.endswith(".nii") or lower.endswith(".nii.gz"):
            yield filename


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
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
        self.transform = transform

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
                for f in _iter_nifti_files(files):
                    seqtype = _seqtype_from_filename(f)
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path)

    def __len__(self):
        return len(self.database)

class BRATSDataset3D(torch.utils.data.Dataset):
    def __init__(self, directory, transform, test_flag=False):
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
        self.transform = transform

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        self.index = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in _iter_nifti_files(files):
                    seqtype = _seqtype_from_filename(f)
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
                slice_count = nibabel.load(datapoint[self.seqtypes[0]]).shape[2]
                self.index.extend((len(self.database) - 1, i) for i in range(slice_count))
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, x):
        out = []
        n, slice = self.index[x]
        filedict = self.database[n]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            o = torch.tensor(nib_img.get_fdata())[:,:,slice]
            # if seqtype != 'seg':
            #     o = o / o.max()
            out.append(o)
        out = torch.stack(out)
        if self.test_flag:
            image=out
            # image = image[..., 8:-8, 8:-8]     #crop to a size of (224, 224)
            if self.transform:
                image = self.transform(image)
            return (image, image, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            # image = image[..., 8:-8, 8:-8]      #crop to a size of (224, 224)
            # label = label[..., 8:-8, 8:-8]
            label=torch.where(label > 0, 1, 0).float()  #merge all tumor classes into one
            if self.transform:
                state = torch.get_rng_state()
                image = self.transform(image)
                torch.set_rng_state(state)
                label = self.transform(label)
            return (image, label, path.split('.nii')[0] + "_slice" + str(slice)+ ".nii") # virtual path


class CachedBRATSSliceDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, test_flag=False, max_open_cases=16):
        super().__init__()
        self.cache_dir = os.path.expanduser(cache_dir)
        self.test_flag = test_flag
        self.max_open_cases = max_open_cases
        metadata_path = os.path.join(self.cache_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"missing BraTS slice cache metadata: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if self.metadata.get("test_flag", False) != test_flag:
            raise ValueError(
                f"cache test_flag={self.metadata.get('test_flag')} does not match requested test_flag={test_flag}"
            )

        self.cases = self.metadata["cases"]
        self.index = []
        for case_idx, case in enumerate(self.cases):
            self.index.extend((case_idx, slice_idx) for slice_idx in range(case["slice_count"]))
        self._case_cache = OrderedDict()

    def __len__(self):
        return len(self.index)

    def _load_case(self, case_idx):
        cached = self._case_cache.get(case_idx)
        if cached is not None:
            self._case_cache.move_to_end(case_idx)
            return cached

        case = self.cases[case_idx]
        image_path = os.path.join(self.cache_dir, case["image_file"])
        image = np.load(image_path, mmap_mode="r")
        label = None
        if not self.test_flag:
            label_path = os.path.join(self.cache_dir, case["label_file"])
            label = np.load(label_path, mmap_mode="r")
        cached = (image, label, case)
        self._case_cache[case_idx] = cached
        if len(self._case_cache) > self.max_open_cases:
            self._case_cache.popitem(last=False)
        return cached

    def __getitem__(self, x):
        case_idx, slice_idx = self.index[x]
        image_arr, label_arr, case = self._load_case(case_idx)
        image = torch.from_numpy(np.array(image_arr[slice_idx], copy=True)).float()

        if self.test_flag:
            source_files = case.get("source_files", {})
            source_path = source_files.get("flair") or source_files.get("t1") or case["case_id"]
            name = source_path.split(".nii")[0] + "_slice" + str(slice_idx) + ".nii"
            return image, image, name

        label = torch.from_numpy(np.array(label_arr[slice_idx], copy=True)).float()
        seg_path = case.get("source_files", {}).get("seg")
        if seg_path:
            name = seg_path.split(".nii")[0] + "_slice" + str(slice_idx) + ".nii"
        else:
            name = f"{case['case_id']}_slice{slice_idx}.nii"
        return image, label, name


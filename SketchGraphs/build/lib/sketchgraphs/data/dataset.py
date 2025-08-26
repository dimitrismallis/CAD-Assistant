import enum
import sketchgraphs.data as datalib
import torch
from sketchgraphs.data import flat_array
import numpy as np
from sketchgraphs.data import Arc, Circle, Line, Point
from sketchgraphs.data import data_utils
from sketchgraphs.data import noise_models
import image_dataset
import os
import PIL
import PIL.Image
import logging
import io
import torchvision
import copy
from torch.nn import functional as F
from sketchgraphs.data import noise_models
from random import shuffle
from collections import OrderedDict
import copy
import torchvision.transforms as T
from sketchgraphs.data.data_utils import NUM_PARAMS
from sketchgraphs.data.sequence import sketch_to_sequence

NON_COORD_TOKEN = 1  # 0 is reserved for padding
CONSTRAINT_COORD_TOKENS = [NON_COORD_TOKEN+1, NON_COORD_TOKEN+2]  # [2, 3]
INCLUDE_CONSTRUCTION = True
COORD_TOKEN_MAP = {}
tok = NON_COORD_TOKEN + 1
for ent_type in [Arc, Circle, Line, Point]:
    COORD_TOKEN_MAP[ent_type] = list(range(tok, tok + NUM_PARAMS[ent_type]))
    tok += NUM_PARAMS[ent_type]

def _pad_or_truncate_to_length(arr, target_length, val=0):
    if target_length is None:
        return arr

    if len(arr) > target_length:
        return arr[:target_length]

    if isinstance(arr, np.ndarray):
        return np.pad(
            arr, (0, target_length - len(arr)), constant_values=val
        )
    elif isinstance(arr, torch.Tensor):
        return torch.nn.functional.pad(
            arr, (0, target_length - len(arr)), value=PrimitiveToken.Pad
        )
    else:
        raise ValueError("arr must be either numpy array or torch Tensor")

class PrimitiveToken(enum.IntEnum):
    """Enumeration indicating the non-parameter value tokens of PrimitiveModel.
    """
    Pad = 0
    Start = 1
    Stop = 2
    Arc = 3
    Circle = 4
    Line = 5
    Point = 6

class ConstraintToken(enum.IntEnum):
    """Enumeration indicating the non-parameter value tokens of ConstraintModel.

    At the moment, only categorical constraints are considered.
    """
    Pad = 0
    Start = 1
    Stop = 2
    Coincident = 3
    Concentric = 4
    Equal = 5
    Fix = 6
    Horizontal = 7
    Midpoint = 8
    Normal = 9
    Offset = 10
    Parallel = 11
    Perpendicular = 12
    Quadrant = 13
    Tangent = 14
    Vertical = 15

class SketchDataset(torch.utils.data.Dataset):
    """Dataset for SketchGraphs primitives."""

    def __init__(
        self,
        sequence_file: str,
        image_data_folder: str,
        num_bins=64,
        max_token_length=130,
        batch_size=256,
        num_workers=8,
        tokenize=True,
        permute_entities=False,
        use_noisy_img=True,
        augmentation_enabled=True,
        use_noisy_prim=False,
        num_points=256,
        ntokens_per_prim=8,
        mask_primitives=False,
        sort_primitives=True,
        sort_parameters=True,
        pad_primitives=True
    ):
        self.sequences = flat_array.load_dictionary_flat(sequence_file)["sequences"]
        self.image_data_folder = image_data_folder
        self.image_dataset = image_dataset.ImageBytesDataset(
            self._list_files_in_directory(self.image_data_folder)
        )
        self.num_bins = num_bins
        self.max_token_length = max_token_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.permute_entities = permute_entities
        self.num_classes = len(PrimitiveToken) + self.num_bins + 2

        if augmentation_enabled == True:
            self.image_transform = torchvision.transforms.RandomAffine(
                4,
                translate=(2 / 128.0, 2 / 128.0),
                scale=(1 - 0.2, 1 + 0.2),
                shear=4,
                fillcolor=1,
            )
        else:
            self.image_transform = None

        self.use_noisy_img = use_noisy_img
        self.use_noisy_prim = use_noisy_prim
        self.pad_primitives = pad_primitives
        self.num_points = num_points
        self.ntokens_per_prim = ntokens_per_prim
        self.mask_primitives = mask_primitives
        self.sort_primitives = sort_primitives
        self.sort_parameters = sort_parameters

    def _list_files_in_directory(self, directory):
        files = [os.path.join(directory, f) for f in os.listdir(directory)]
        files.sort()
        return files

    def apply_permute_entities(self, sketch):
        # Randomly permute primitive ordering
        entities = list(sketch.entities.items())
        shuffle(entities)
        sketch.entities = OrderedDict(entities)
        return sketch

    # def fix_line_direction(self, sketch):
    #     entities = list(sketch.entities.values())
    #     new_ents = []
    #     # pdb.set_trace()
    #     for entity in entities:
    #         if entity.type.name == "Line":
    #             pdb.set_trace()
    #         new_ents.append(entity)
    #     sketch.entities = OrderedDict(entities)

    def tokenize_primitives(
        self,
        sketch: datalib.Sketch,
        num_bins: int,
        max_length=None,
        permute=False,
        include_stop=True,
        pad_primitives=True,
    ):
        val_tokens = [PrimitiveToken.Start]
        coord_tokens = [NON_COORD_TOKEN]
        pos_idx = 1  # 0 is reserved for padding
        pos_tokens = [pos_idx]

        # isConstruction tokens
        construction_tok_dict = {
            True: len(PrimitiveToken) + num_bins,
            False: len(PrimitiveToken) + num_bins + 1,
        }

        if permute:
            sketch = self.apply_permute_entities(sketch)

        ent_values = list(sketch.entities.values())

        # Index tracking for constraint model's gather operation
        gather_map = {Arc: [0, 1, 3, 5], Circle: [0, 1], Line: [0, 1, 3], Point: [0]}
        gather_idxs = [0]  # 0 is for external even though we don't use

        for ent in ent_values:
            num_val = len(val_tokens)
            if ent is not None:
                gather_idxs.extend(
                    [len(val_tokens) + gidx for gidx in gather_map[type(ent)]]
                )

                val_tokens.append(PrimitiveToken[ent.type.name])
                coord_tokens.append(NON_COORD_TOKEN)
                pos_idx += 1
                pos_tokens.append(pos_idx)
                params, _ = data_utils.parameterize_entity(ent, self.sort_parameters)
                try:
                    param_bins = data_utils.quantize_params(params, num_bins)
                except:
                    params[params > 0.5] = 0.5
                    params[params < -0.5] = -0.5
                    param_bins = data_utils.quantize_params(params, num_bins)

                val_tokens.extend(param_bins + len(PrimitiveToken))
                coord_tokens.extend(COORD_TOKEN_MAP[type(ent)])
                pos_tokens.extend([pos_idx] * param_bins.size)

                # Add isConstruction attribute
                if INCLUDE_CONSTRUCTION:
                    val_tokens.append(construction_tok_dict[ent.isConstruction])
                    coord_tokens.append(NON_COORD_TOKEN)
                    pos_tokens.append(pos_idx)

            if pad_primitives == True:
                num_paddings = num_val + self.ntokens_per_prim - len(val_tokens)
                if num_paddings > 0:
                    for k in range(num_paddings):
                        val_tokens.append(0)
                        pos_tokens.append(pos_tokens[-1])
                        coord_tokens.append(coord_tokens[-1])

        if include_stop:
            val_tokens.append(PrimitiveToken.Stop)
            coord_tokens.append(NON_COORD_TOKEN)
            pos_tokens.append(pos_idx + 1)

        val_token_types = _pad_or_truncate_to_length(
            np.array(val_tokens, dtype=np.int64), max_length
        )
        val_token_types[
            np.logical_and(
                val_token_types >= len(PrimitiveToken),
                (val_token_types < len(PrimitiveToken) + num_bins),
            )
        ] = PrimitiveToken.Pad

        sample = {
            "val": _pad_or_truncate_to_length(
                np.array(val_tokens, dtype=np.int64), max_length
            ),
            "coord": _pad_or_truncate_to_length(
                np.array(coord_tokens, dtype=np.int64), max_length
            ),
            "pos": _pad_or_truncate_to_length(
                np.array(pos_tokens, dtype=np.int64), max_length
            ),
        }

        return sample, gather_idxs

    def tokenize_sequence(self, seq):
        """Tokenization of a sketch sequence with primitives and constraints."""
        sketch = datalib.sketch_from_sequence(seq)
        return self.tokenize_sketch(sketch)
    
    def tokenize_constraints(self, seq, gather_idxs, max_length=None):
        val_tokens = [ConstraintToken.Start]
        coord_tokens = [NON_COORD_TOKEN]
        pos_idx = 1  # 0 is reserved for padding
        pos_tokens = [pos_idx]

        # Iterate through edge ops
        for op in seq:
            # Ensure op is applicable edge op
            if not isinstance(op, datalib.EdgeOp):
                continue
            if not op.label.name in ConstraintToken.__members__:
                continue
            refs = op.references
            if 0 in refs:  # skip external constraints
                continue

            num_val = len(val_tokens)

            # Add constraint type tokens
            val_tokens.append(ConstraintToken[op.label.name])
            coord_tokens.append(NON_COORD_TOKEN)
            pos_idx += 1
            pos_tokens.append(pos_idx)

            # Add reference parameters
            val_tokens.extend(
                [gather_idxs[ref] + len(ConstraintToken) for ref in sorted(refs)]
            )
            coord_tokens.extend(CONSTRAINT_COORD_TOKENS[: len(refs)])
            pos_tokens.extend([pos_idx] * len(refs))

            # # pdb.set_trace()
            # if pad_constraints == True:
            #     num_paddings = num_val + 8 - len(val_tokens)
            #     if num_paddings > 0:
            #         for k in range(num_paddings):
            #             val_tokens.append(0)
            #             pos_tokens.append(1)
            #             coord_tokens.append(NON_COORD_TOKEN)

        val_tokens.append(ConstraintToken.Stop)
        coord_tokens.append(NON_COORD_TOKEN)
        pos_tokens.append(pos_idx + 1)

        sample = {
            "val": _pad_or_truncate_to_length(
                np.array(val_tokens, dtype=np.int64), max_length
            ),
            "coord": _pad_or_truncate_to_length(
                np.array(coord_tokens, dtype=np.int64), max_length
            ),
            "pos": _pad_or_truncate_to_length(
                np.array(pos_tokens, dtype=np.int64), max_length
            ),
        }

        return sample


    def tokenize_sketch(self, sketch):
        """Tokenization of a sketch with primitives and constraints."""
        if self.use_noisy_prim == True:
            sketch = self.apply_primitive_noise(sketch)
        prim_tokens, gather_idx = self.tokenize_primitives(
            sketch,
            num_bins=self.num_bins,
            max_length=self.max_token_length,
            permute=self.permute_entities,
            pad_primitives=self.pad_primitives,
        )
        seq = sketch_to_sequence(sketch)
        constraint_tokens = self.tokenize_constraints(
            seq, gather_idx, self.max_token_length
        )
        return prim_tokens, constraint_tokens, gather_idx

    def tokens_to_tensor(self, prim_tokens, constraint_tokens):
        sample = {}
        for k, v in prim_tokens.items():
            sample[k] = torch.from_numpy(prim_tokens[k])
        if constraint_tokens != None:
            for k, v in constraint_tokens.items():
                sample[f"c_{k}"] = torch.from_numpy(constraint_tokens[k])
        return sample

    def get_image(self, idx, use_noisy_img=False, binarize_img=False):
        image_bytes_arr = self.image_dataset[idx]
        try:
            if len(image_bytes_arr) > 1 and use_noisy_img == True:
                image_bytes = image_bytes_arr[
                    torch.randint(1, len(image_bytes_arr), ())
                ]
            else:
                image_bytes = image_bytes_arr[0]

            img = PIL.Image.open(io.BytesIO(image_bytes)).convert("L")
            if binarize_img == True:
                img = self.binarize_image(img)
            img_tensor = torchvision.transforms.ToTensor()(img)

            if (self.image_transform is not None) and use_noisy_img == True:
                img_tensor = self.image_transform(img_tensor)

        except PIL.UnidentifiedImageError:
            logging.getLogger(__name__).warn(
                "Failed to decode image at index {}".format(idx)
            )
            img_tensor = torch.zeros((1, 128, 128))

        return img_tensor

    def get_sub_image(self, idx, subimage_idx, binarize_img=False):
        image_bytes_arr = self.image_dataset[idx]
        # import pdb; pdb.set_trace()
        # print(len(image_bytes_arr))
        try:
            image_bytes = image_bytes_arr[subimage_idx]

            img = PIL.Image.open(io.BytesIO(image_bytes)).convert("L")
            if binarize_img == True:
                img = self.binarize_image(img)
            img_tensor = torchvision.transforms.ToTensor()(img)

            if self.image_transform is not None:
                img_tensor = self.image_transform(img_tensor)

        except PIL.UnidentifiedImageError:
            logging.getLogger(__name__).warn(
                "Failed to decode image at index {}".format(idx)
            )
            img_tensor = torch.zeros((1, 128, 128))

        return img_tensor

    def apply_primitive_noise(
        self, sketch, std=0.15, max_difference=0.15
    ) -> datalib.Sketch:
        noise_sketch = copy.deepcopy(sketch)
        try:
            noise_models.noisify_sketch_ents(
                noise_sketch, std=std, max_diff=max_difference
            )
        except:
            noise_sketch = sketch
        return noise_sketch

    def binarize_image(self, img, threshold=127):
        arr = np.array(img)
        arr[arr <= threshold] = 0
        arr[arr > threshold] = 255
        return PIL.Image.fromarray(arr)

    def sketch_from_idx(self, idx):
        seq = self.sequences[idx]
        sketch = datalib.sketch_from_sequence(seq)
        return sketch

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        sketch = datalib.sketch_from_sequence(seq)
        prim_tokens, constraint_tokens, gather_idxs = self.tokenize_sketch(sketch)
        sample = self.tokens_to_tensor(prim_tokens, constraint_tokens)

        img = self.get_image(idx, binarize_img=False)

        sample["img"] = img
        sample["hand_drawn"] = self.get_image(idx, use_noisy_img=True)
        sample["idx"] = idx

        if self.use_noisy_img == True:
            sample["img"] = sample["hand_drawn"]

        return sample

    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )

    def __len__(self):
        return len(self.sequences)
    


def param_seq_from_tokens(tokens, num_bins):
    reverse_construction_toks = {
        len(PrimitiveToken) + num_bins: True,
        len(PrimitiveToken) + num_bins + 1: False,
    }

    all_params = []
    curr_params = []
    for token in tokens:
        if token == PrimitiveToken.Start:
            continue
        if token < len(PrimitiveToken):
            if curr_params:
                all_params.append((curr_params, isConstruction))
                curr_params = []
        if token in [
            PrimitiveToken.Stop,
            PrimitiveToken.Pad,
        ]:
            break
        if token >= len(PrimitiveToken):
            isConstruction = False  # initialize to False in case not modeling
            if token <= len(PrimitiveToken) + (num_bins - 1):
                # Numerical coordinate
                curr_params.append(token - len(PrimitiveToken))
            else:
                # isConstruction attribute
                isConstruction = reverse_construction_toks[token]

    if curr_params:
        # Append possibly leftover entity parameters
        all_params.append((curr_params, isConstruction))
    return all_params


def sketch_from_tokens(tokens, num_bins):
    if type(tokens) == torch.Tensor:
        tokens = tokens.long().detach().cpu().numpy()
    all_params = param_seq_from_tokens(tokens, num_bins)
    sketch = datalib.Sketch()
    for idx, (ent_params, isConstruction) in enumerate(all_params):
        ent_params = data_utils.dequantize_params(ent_params, num_bins)
        ent_params = ent_params.tolist()
        try:
            ent = data_utils.entity_from_params(ent_params)
        except:
            ent = None
        if ent is not None:
            ent.entityId = str(idx + 1)
            ent.isConstruction = isConstruction
            sketch.entities[str(idx)] = ent

    return sketch


def sketch_from_constraint_tokens(tokens, sketch: datalib.Sketch, gather_idxs):
    """Add constraint value tokens to Sketch instance.
    """
    seq = datalib.sketch_to_sequence(sketch)
    stop_node = seq.pop()  # add back afterwards

    reverse_gather = {val:idx for idx, val in enumerate(gather_idxs)}
    new_type = None
    new_refs = []
    for token in tokens:
        if token < len(ConstraintToken):
            # Add previous completed constraint, if any
            if new_type is not None:
                new_op = datalib.EdgeOp(
                    label=new_type, references=tuple(new_refs))
                new_type = None
                new_refs = []
                seq.append(new_op)
            if token <= ConstraintToken.Stop:
                continue
            # Get constraint type
            new_type = datalib.ConstraintType[ConstraintToken(token).name]
        else:
            ori_token = token - len(ConstraintToken)
            new_refs.append(reverse_gather[ori_token])

    seq.append(stop_node)
    return datalib.sketch_from_sequence(seq)
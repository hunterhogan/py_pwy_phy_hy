"""Access PyTorch tensor-shaping, masking, padding, and checkpoint utilities.

You can use this package for optional-value handling, tensor slicing, rank alignment, mask
construction, safe concatenation, sequence padding, PyTree traversal, and `einops`-pattern tensor
packing.

Helpers
-------
compact
	Filter `None` values from an iterable and return a `list` of the remaining values.
default
	Return a fallback value when the primary value is `None`.
divisible_by
	Test whether one numeric value is evenly divisible by another.
exists
	Test whether a value is not `None`.
first
	Return the element at index `0` of an integer-indexable object.
identity
	Return the first argument unchanged while ignoring all other arguments.
map_values
	Apply a function to every leaf value in a nested `list`, `tuple`, or `dict` structure.
maybe
	Wrap a callable so it skips work when the first argument is `None`.
once
	Wrap a callable so the callable executes at most once.
safe
	Wrap a callable so `None` values are removed from the first tensor sequence argument.

Slicing and reshaping
---------------------
shape_with_replace
	Compute a derived shape by replacing selected dimension sizes.
slice_at_dim
	Apply an arbitrary `slice` to one tensor dimension.
slice_left_at_dim
	Select a prefix of a tensor dimension.
slice_right_at_dim
	Select a suffix of a tensor dimension.

Adding dimensions
-----------------
align_dims_left
	Pad trailing singleton dimensions across several tensors to a shared rank.
pad_left_ndim
	Prepend a fixed number of singleton dimensions.
pad_left_ndim_to
	Prepend singleton dimensions until a tensor reaches a target rank.
pad_ndim
	Insert singleton dimensions on both sides of a tensor shape.
pad_right_ndim
	Append a fixed number of singleton dimensions.
pad_right_ndim_to
	Append singleton dimensions until a tensor reaches a target rank.

Mask tools
----------
and_masks
	Combine boolean masks with element-wise logical AND.
lens_to_mask
	Convert length values into boolean masks.
or_masks
	Combine boolean masks with element-wise logical OR.
reduce_masks
	Reduce a sequence of boolean masks with a caller-supplied binary operator.

Concatenating and stacking
--------------------------
safe_cat
	Concatenate tensors while skipping `None` values.
safe_stack
	Stack tensors while skipping `None` values.

Padding
-------
pad_at_dim
	Pad or trim a tensor along one dimension.
pad_left_at_dim
	Pad the left side of one tensor dimension by an explicit count.
pad_left_at_dim_to
	Pad the left side of one tensor dimension until it reaches a target length.
pad_right_at_dim
	Pad the right side of one tensor dimension by an explicit count.
pad_right_at_dim_to
	Pad the right side of one tensor dimension until it reaches a target length.
pad_sequence
	Pad a sequence of tensors to a shared length and optionally stack them.
pad_sequence_and_cat
	Pad a sequence of tensors to a shared length and concatenate the padded tensors.

Utilities
---------
l2norm
	Normalize `Tensor` vectors to unit length along the last dimension.
masked_mean
	Compute a mean over positions selected by a boolean mask.
tree_flatten_with_inverse
	Flatten a PyTree and return an inverse reconstruction function.
tree_map_tensor
	Apply a function to every tensor leaf in a PyTree.

Modules
-------
device
	Determine `torch.nn.Module` devices and decorate callables to move `Tensor` arguments automatically.
einops
	Pack and unpack `Tensor` objects with `einops` patterns and paired inverse functions.
save_load
	Decorate `torch.nn.Module` subclasses with checkpoint save, load, and reconstruction helpers.
"""
# isort: split
from py_pwy_phy_hy._semiotics import decreasing as decreasing, zeroIndexed as zeroIndexed

# isort: split
from py_pwy_phy_hy._types import (
	ConfigArgsKwargs as ConfigArgsKwargs, DehydratedCheckpoint as DehydratedCheckpoint, DehydratedTorchNNModule as DehydratedTorchNNModule,
	DimAndValue as DimAndValue, IdentityCallable as IdentityCallable, PSpec as PSpec, RVar as RVar, StrPath as StrPath,
	SupportsIntIndex as SupportsIntIndex, T_co as T_co, TorchNNModule as TorchNNModule, TVar as TVar)

# isort: split
from py_pwy_phy_hy._helpers import (
	compact as compact, default as default, divisible_by as divisible_by, exists as exists, first as first, identity as identity,
	map_values as map_values, maybe as maybe, once as once, safe as safe)

# isort: split
from py_pwy_phy_hy._slicing import (
	shape_with_replace as shape_with_replace, slice_at_dim as slice_at_dim, slice_left_at_dim as slice_left_at_dim,
	slice_right_at_dim as slice_right_at_dim)

# isort: split
from py_pwy_phy_hy._dimensions import (
	align_dims_left as align_dims_left, pad_left_ndim as pad_left_ndim, pad_left_ndim_to as pad_left_ndim_to, pad_ndim as pad_ndim,
	pad_right_ndim as pad_right_ndim, pad_right_ndim_to as pad_right_ndim_to)

# isort: split
from py_pwy_phy_hy._masking import and_masks as and_masks, lens_to_mask as lens_to_mask, or_masks as or_masks, reduce_masks as reduce_masks

# isort: split
from py_pwy_phy_hy._cat_and_stack import safe_cat as safe_cat, safe_stack as safe_stack

# isort: split
from py_pwy_phy_hy._padding import (
	pad_at_dim as pad_at_dim, pad_left_at_dim as pad_left_at_dim, pad_left_at_dim_to as pad_left_at_dim_to,
	pad_right_at_dim as pad_right_at_dim, pad_right_at_dim_to as pad_right_at_dim_to, pad_sequence as pad_sequence,
	pad_sequence_and_cat as pad_sequence_and_cat)

# isort: split
from py_pwy_phy_hy.utils import l2norm as l2norm, masked_mean as masked_mean

# isort: split
from py_pwy_phy_hy.utils import tree_flatten_with_inverse as tree_flatten_with_inverse, tree_map_tensor as tree_map_tensor

# isort: split
# NOTE encourage users to import pack_with_inverse from the einops submodule for discoverability
from py_pwy_phy_hy.einops import pack_with_inverse  # pyright: ignore[reportUnusedImport]

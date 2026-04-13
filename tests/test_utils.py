from collections.abc import Sequence
from py_pwy_phy_hy import (
	align_dims_left, and_masks, lens_to_mask, maybe, or_masks, pad_at_dim, pad_left_at_dim, pad_left_at_dim_to, pad_left_ndim_to, pad_ndim,
	pad_right_at_dim, pad_right_at_dim_to, pad_right_ndim_to, pad_sequence, pad_sequence_and_cat, safe_cat, safe_stack, shape_with_replace,
	slice_at_dim, slice_left_at_dim, slice_right_at_dim)
from py_pwy_phy_hy.einops import pack_with_inverse
from py_pwy_phy_hy.utils import exists, masked_mean, tree_flatten_with_inverse, tree_map_tensor
from torch import tensor
import torch

def test_exist() -> None:
	assert not exists(None)

def test_maybe() -> None:
	assert maybe(None)(1) == 1
	assert not exists(maybe(lambda t: t + 1)(None))

def test_pad_ndim() -> None:
	t = torch.randn(3)
	t = pad_ndim(t, (1, 2))
	assert t.shape == (1, 3, 1, 1)

	t = torch.randn(3)
	t = pad_right_ndim_to(t, 3)
	assert t.shape == (3, 1, 1)

	t = torch.randn(3, 4, 5)
	t = pad_right_ndim_to(t, 3)
	assert t.shape == (3, 4, 5)

	t = torch.randn(3)
	t = pad_left_ndim_to(t, 3)
	assert t.shape == (1, 1, 3)

def test_align_ndim_left() -> None:
	t = torch.randn(3)
	u = torch.randn(3, 5, 2)
	v = torch.randn(3, 5)

	t, u, v = align_dims_left((t, u, v))
	assert t.shape == (3, 1, 1)
	assert u.shape == (3, 5, 2)
	assert v.shape == (3, 5, 1)

def test_pad_at_dim() -> None:
	t = torch.randn(3, 6, 1)
	padded = pad_at_dim(t, (0, 1), dim = 1)

	assert padded.shape == (3, 7, 1)
	assert torch.allclose(padded, pad_right_at_dim(t, 1, dim = 1))
	assert not torch.allclose(padded, pad_left_at_dim(t, 1, dim = 1))

	t = torch.randn(3, 6, 1)
	padded = pad_right_at_dim_to(t, 7, dim = 1)
	assert padded.shape == (3, 7, 1)

	padded = pad_left_at_dim_to(t, 7, dim = 1)
	assert padded.shape == (3, 7, 1)

	padded = pad_right_at_dim_to(t, 6, dim = 1)
	assert padded.shape == (3, 6, 1)

def test_tree_flatten_with_inverse() -> None:
	tree = (1, (2, 3), 4)
	(first, *rest), inverse = tree_flatten_with_inverse(tree)

	out = inverse((first + 1, *rest))
	assert out == (2, (2, 3), 4)

def test_tree_map_tensor()	-> None:
	tree = (1, tensor(2), 3)
	tree = tree_map_tensor(lambda t: t + 1, tree)
	assert tree[0] == 1
	assert tree[-1] == 3
	assert (tree[1] == 3).all()

def test_pack_with_inverse() -> None:
	t: torch.Tensor = torch.randn(3, 12, 2, 2)
	t, inverse = pack_with_inverse(t, 'b * d')

	assert t.shape == (3, 24, 2)
# TODO look at this return type.
	t = inverse(t)
	assert t.shape == (3, 12, 2, 2)

	u = torch.randn(3, 4, 2)
	t, inverse = pack_with_inverse([t, u], 'b * d')
	assert t.shape == (3, 28, 2)

	t = t.sum(dim = -1)
	t, u = inverse(t, 'b *')
	assert t.shape == (3, 12, 2)
	assert u.shape == (3, 4)

def test_better_pad_sequence() -> None:

	x: torch.Tensor = torch.randn(2, 4, 5)
	y: torch.Tensor = torch.randn(2, 3, 5)
	z: torch.Tensor = torch.randn(2, 1, 5)
	tensors: Sequence[torch.Tensor] = [x, y, z]

	packed, lens = pad_sequence(tensors=tensors, dim = 1, return_lens = True)
	assert packed.shape == (3, 2, 4, 5)
	assert lens.tolist() == [4, 3, 1] # pyright: ignore[reportUnknownMemberType]

	mask = lens_to_mask(lens)
	assert torch.allclose(mask.sum(dim = -1), lens)

def test_pad_sequence_uneven_images() -> None:
	images = [
		torch.randn(3, 16, 17),
		torch.randn(3, 15, 18),
		torch.randn(3, 17, 16)
	]

	padded_height = pad_sequence(images, dim = -2, return_stacked = False)
	assert len(padded_height) == 3
	assert all(t.shape[1] == 17 for t in padded_height)

	stacked = pad_sequence_and_cat(padded_height, dim_cat = 0)
	assert stacked.shape == (9, 17, 18)

def test_and_masks() -> None:
	assert not exists(and_masks([None]))

	mask1 = tensor([True, True])
	mask2 = tensor([True, False])
	assert (and_masks([mask1, None, mask2]) == tensor([True, False])).all()

def test_or_masks() -> None:
	assert not exists(or_masks([None]))

	mask1 = tensor([True, True])
	mask2 = tensor([True, False])
	assert (or_masks([mask1, None, mask2]) == tensor([True, True])).all()

def test_masked_mean() -> None:
	t = tensor([1., 2., 3., 4.])
	assert torch.allclose(masked_mean(t), tensor(2.5))
	assert torch.allclose(masked_mean(t, dim = 0), tensor(2.5))

	mask = tensor([True, False, True, False])
	assert torch.allclose(masked_mean(t, mask = mask), tensor(2.0))

	mask = tensor([False, False, False, False])
	assert torch.allclose(masked_mean(t, mask = mask), tensor(0.0))

	t = tensor([[1., 2.], [3., 4.]])
	mask = tensor([[True, False], [True, True]])

	assert torch.allclose(masked_mean(t, mask = mask, dim = 0), tensor([2.0, 4.0]))

	assert torch.allclose(masked_mean(t, mask = mask, dim = 1), tensor([1.0, 3.5]))

	t = torch.randn(2, 3, 4)
	mask = torch.ones(2, 3, 4).bool()
	mask[0, :, :] = False

	res = masked_mean(t, mask = mask, dim = (1, 2))
	assert res.shape == (2,)
	assert torch.allclose(res[0], tensor(0.0), atol = 1e-4)
	assert torch.allclose(res[1], t[1].mean())

	t = torch.randn(2, 3, 4)
	mask = tensor([True, False])
	res = masked_mean(t, mask = mask, dim = (1, 2))
	assert res.shape == (2,)
	assert torch.allclose(res[0], t[0].mean())
	assert torch.allclose(res[1], tensor(0.0), atol = 1e-4)

def test_slice_at_dim() -> None:
	t = torch.randn(3, 4, 5)

	res = slice_at_dim(t, slice(1, 3))
	assert res.shape == (3, 4, 2)
	assert torch.allclose(res, t[:, :, 1:3])

	res = slice_at_dim(t, slice(None, 2), dim = 1)
	assert res.shape == (3, 2, 5)
	assert torch.allclose(res, t[:, :2, :])

	res = slice_at_dim(t, slice(2, None), dim = -2)
	assert res.shape == (3, 2, 5)
	assert torch.allclose(res, t[:, 2:, :])

	res = slice_left_at_dim(t, 2, dim = 1)
	assert res.shape == (3, 2, 5)
	assert torch.allclose(res, t[:, :2, :])

	res = slice_right_at_dim(t, 2, dim = 1)
	assert res.shape == (3, 2, 5)
	assert torch.allclose(res, t[:, -2:, :])

def test_shape_with_replace() -> None:
	t = torch.randn(3, 4, 5)
	assert shape_with_replace(t, {1: 2}) == (3, 2, 5)

def test_safe_functions() -> None:
	t1 = torch.randn(2, 3)
	t2 = torch.randn(2, 3)

	assert safe_stack([]) is None
	assert safe_stack([None]) is None
	assert (safe_stack([t1]) == t1).all()
	assert (safe_stack([t1, None]) == t1).all()
	assert safe_stack([t1]).shape == (1, 2, 3)
	assert safe_stack([t1, t2]).shape == (2, 2, 3)

	assert safe_cat([]) is None
	assert safe_cat([None]) is None
	assert (safe_cat([t1]) == t1).all()
	assert (safe_cat([t1, None]) == t1).all()
	assert safe_cat([t1, t2]).shape == (4, 3)

"""
Some or all of the logic in this module may be protected by the following.

MIT License

Copyright (c) 2026 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

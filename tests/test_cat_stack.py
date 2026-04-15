from __future__ import annotations

from collections.abc import Sequence
from py_pwy_phy_hy import broadcast_cat, safe_cat, safe_stack
from torch import Tensor
import pytest
import torch

def test_safe_stack(sequence_collection: list[Sequence[Tensor]]) -> None:
	for sequence in sequence_collection:
		unique_shapes = {t.shape for t in sequence}
		if len(unique_shapes) != 1:
			continue
		result = safe_stack(sequence)
		assert isinstance(result, Tensor), (
			f"safe_stack returned {type(result).__name__}, expected Tensor "
			f"for sequence of {len(sequence)} tensors each with shape {sequence[0].shape}."
		)
		assert result.shape[0] == len(sequence), (
			f"safe_stack returned shape {result.shape}, expected dim 0 to be {len(sequence)} "
			f"for {len(sequence)} input tensors with shape {sequence[0].shape}."
		)
		assert result.shape[1:] == sequence[0].shape, (
			f"safe_stack returned shape {result.shape}, expected trailing shape {sequence[0].shape} "
			f"matching input tensor shape for sequence of length {len(sequence)}."
		)

def test_safe_cat(sequence_collection: list[Sequence[Tensor]]) -> None:
	for sequence in sequence_collection:
		unique_ranks = {t.dim() for t in sequence}
		unique_trailing_shapes = {t.shape[1:] for t in sequence}
		if len(unique_ranks) != 1 or len(unique_trailing_shapes) != 1:
			continue
		result = safe_cat(sequence)
		expected_dim0 = sum(t.shape[0] for t in sequence)
		assert isinstance(result, Tensor), (
			f"safe_cat returned {type(result).__name__}, expected Tensor for sequence of {len(sequence)} tensors."
		)
		assert result.shape[0] == expected_dim0, (
			f"safe_cat returned shape {result.shape}, expected dim 0 to be {expected_dim0} "
			f"from concatenating {len(sequence)} tensors along dim 0."
		)
		assert result.shape[1:] == sequence[0].shape[1:], (
			f"safe_cat returned shape {result.shape}, expected trailing shape {sequence[0].shape[1:]} "
			f"matching input tensor trailing shape for sequence of length {len(sequence)}."
		)

@pytest.mark.parametrize(
	("singleton_axes", "dim"),
	[
		pytest.param((0,), 0, id='singleton-leading-cat-leading'),
		pytest.param((-1,), -1, id='singleton-trailing-cat-trailing'),
		pytest.param((0, -1), 0, id='singleton-leading-trailing-cat-leading'),
	],
)
def test_broadcast_cat(t: Tensor, singleton_axes: tuple[int, ...], dim: int) -> None:
	normalized_axes = {axis if axis >= 0 else t.ndim + axis for axis in singleton_axes}
	broadcast_source = t[
		tuple(
			slice(0, 1) if axis_index in normalized_axes else slice(None)
			for axis_index in range(t.ndim)
		)
	]
	result = broadcast_cat((t, broadcast_source), dim=dim)
	expected = torch.cat((t, broadcast_source.expand_as(t)), dim=dim)
	normalized_dim = dim if dim >= 0 else t.ndim + dim
	expected_shape = list(t.shape)
	expected_shape[normalized_dim] *= 2

	assert broadcast_source.ndim == t.ndim, (
		f"broadcast_cat test constructed rank {broadcast_source.ndim} tensor, expected rank {t.ndim} "
		f"for input shape {tuple(t.shape)} and {singleton_axes=}."
	)
	assert result.shape == torch.Size(expected_shape), (
		f"broadcast_cat returned shape {tuple(result.shape)}, expected {tuple(expected_shape)} "
		f"for input shape {tuple(t.shape)}, broadcast source shape {tuple(broadcast_source.shape)}, "
		f"and {dim=}."
	)
	assert torch.equal(result, expected), (
		f"broadcast_cat returned values {result} that do not match explicit broadcast-plus-cat result {expected} "
		f"for input shape {tuple(t.shape)}, broadcast source shape {tuple(broadcast_source.shape)}, and {dim=}."
	)

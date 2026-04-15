from __future__ import annotations

from py_pwy_phy_hy import tree_flatten_with_inverse, tree_map_tensor
from torch.utils._pytree import tree_flatten, tree_unflatten
import pytest
import torch

def _assert_pytree_leaf_values_equal(
	expected_tree: object, actual_tree: object, context_label: str
) -> None:
	expected_flattened, expected_spec = tree_flatten(expected_tree)
	actual_flattened, actual_spec = tree_flatten(actual_tree)

	assert actual_spec == expected_spec, (
		f"PyTree structure mismatch for {context_label}: got spec {actual_spec}, expected {expected_spec}."
	)
	assert len(actual_flattened) == len(expected_flattened), (
		f"PyTree leaf count mismatch for {context_label}: got {len(actual_flattened)}, expected {len(expected_flattened)}."
	)

	for leaf_index, (expected_leaf, actual_leaf) in enumerate(
		zip(expected_flattened, actual_flattened, strict=True)
	):
		if torch.is_tensor(expected_leaf):
			assert torch.is_tensor(actual_leaf), (
				f"PyTree leaf type mismatch at index {leaf_index} for {context_label}: expected tensor leaf."
			)
			assert torch.equal(actual_leaf, expected_leaf), (
				f"PyTree tensor leaf mismatch at index {leaf_index} for {context_label}."
			)
		else:
			assert actual_leaf == expected_leaf, (
				f"PyTree non-tensor leaf mismatch at index {leaf_index} for {context_label}: "
				f"got {actual_leaf!r}, expected {expected_leaf!r}."
			)

@pytest.mark.parametrize(
	("scale", "offset"), [pytest.param(2.0, 3.0, id="scale-two-offset-three")]
)
def test_tree_map_tensor_transforms_only_tensor_leaves(
	tensor_pytree: object,
	scale: float,
	offset: float,
) -> None:
	input_flattened, input_spec = tree_flatten(tensor_pytree)
	tensor_leaf_count = sum(torch.is_tensor(leaf) for leaf in input_flattened)
	non_tensor_leaf_count = len(input_flattened) - tensor_leaf_count

	assert tensor_leaf_count > 0, (
		"tree_map_tensor test input must contain at least one tensor leaf."
	)
	assert non_tensor_leaf_count > 0, (
		"tree_map_tensor test input must contain at least one non-tensor leaf."
	)

	expected_flattened = [
		leaf.to(dtype=torch.float64) * scale + offset if torch.is_tensor(leaf) else leaf
		for leaf in input_flattened
	]
	expected_tree = tree_unflatten(expected_flattened, input_spec)

	mapped_tree = tree_map_tensor(
		lambda tensor_value: tensor_value.to(dtype=torch.float64) * scale + offset,
		tensor_pytree,
	)

	_assert_pytree_leaf_values_equal(
		expected_tree, mapped_tree, "tree_map_tensor affine transform"
	)

@pytest.mark.parametrize("tensor_shift", [pytest.param(5.0, id="tensor-shift-five")])
def test_tree_flatten_with_inverse_reconstructs_and_replaces_leaves(
	tensor_pytree: object,
	tensor_shift: float,
) -> None:
	original_flattened, original_spec = tree_flatten(tensor_pytree)
	flattened, inverse = tree_flatten_with_inverse(tensor_pytree)

	assert len(flattened) == len(original_flattened), (
		f"tree_flatten_with_inverse returned {len(flattened)} leaves, expected {len(original_flattened)}."
	)

	reconstructed_tree = inverse(flattened)
	_assert_pytree_leaf_values_equal(
		tensor_pytree, reconstructed_tree, "tree_flatten_with_inverse round trip"
	)

	shifted_flattened = [
		leaf.to(dtype=torch.float64) + tensor_shift if torch.is_tensor(leaf) else leaf
		for leaf in flattened
	]
	shifted_tree = inverse(shifted_flattened)
	expected_shifted_tree = tree_unflatten(shifted_flattened, original_spec)

	_assert_pytree_leaf_values_equal(
		expected_shifted_tree, shifted_tree, "tree_flatten_with_inverse shifted leaves"
	)


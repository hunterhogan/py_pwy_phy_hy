from __future__ import annotations

from py_pwy_phy_hy import l2norm, masked_mean, tree_flatten_with_inverse, tree_map_tensor
from torch import Tensor
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


@pytest.mark.parametrize("eps", [pytest.param(1e-5, id="eps-default")])
def test_masked_mean_matches_reference(
    t: Tensor,
    boolean_mask_like_t: Tensor,
    reduction_dim: int | None,
    eps: float,
) -> None:
    tensor_value = t.to(dtype=torch.float64)
    mask_value = boolean_mask_like_t

    expanded_mask = mask_value
    if expanded_mask.ndim < tensor_value.ndim:
        expanded_mask = expanded_mask.reshape(
            (*expanded_mask.shape, *(1,) * (tensor_value.ndim - expanded_mask.ndim))
        )
    expanded_mask = expanded_mask.expand_as(tensor_value)

    if reduction_dim is None:
        selected_values = tensor_value[expanded_mask]
        expected = (
            selected_values.mean()
            if bool(expanded_mask.any())
            else selected_values.sum()
        )
    else:
        numerator = (tensor_value * expanded_mask).sum(dim=reduction_dim)
        denominator = expanded_mask.sum(dim=reduction_dim)
        expected = numerator / denominator.clamp(min=eps)

    result = masked_mean(tensor_value, mask=mask_value, dim=reduction_dim, eps=eps)

    assert result.shape == expected.shape, (
        f"masked_mean returned shape {tuple(result.shape)}, expected {tuple(expected.shape)} "
        f"for {tuple(t.shape)=}, {tuple(mask_value.shape)=}, and {reduction_dim=}."
    )
    assert torch.allclose(result, expected), (
        f"masked_mean returned values {result} that do not match expected {expected} "
        f"for {tuple(t.shape)=}, {tuple(mask_value.shape)=}, and {reduction_dim=}."
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


@pytest.mark.parametrize(
    "tolerance",
    [pytest.param(1e-5, id="tolerance-1e-5")],
)
def test_l2norm_produces_unit_length_vectors(
    t: Tensor,
    tolerance: float,
) -> None:
    inputTensor = t.to(dtype=torch.float64)
    resultTensor = l2norm(inputTensor)

    assert resultTensor.shape == inputTensor.shape, (
        f"l2norm returned shape {tuple(resultTensor.shape)}, expected {tuple(inputTensor.shape)} "
        f"for input shape {tuple(t.shape)}."
    )

    vectorNorms = resultTensor.norm(dim=-1)
    expectedNorms = torch.ones_like(vectorNorms)
    assert torch.allclose(vectorNorms, expectedNorms, atol=tolerance), (
        f"l2norm produced non-unit vector norms {vectorNorms} for input shape {tuple(t.shape)} "
        f"with {tolerance=}."
    )



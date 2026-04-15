from __future__ import annotations

from py_pwy_phy_hy.scaleValues import l2norm, masked_mean, RMSNorm
from torch import Tensor
import pytest
import torch

@pytest.mark.parametrize(
	"tolerance",
	[pytest.param(1e-5, id="tolerance-1e-5")],
)
def test_l2norm(t: Tensor, tolerance: float ) -> None:
	inputTensor = t.to(dtype=torch.float64)
	resultTensor = l2norm(inputTensor)

	assert resultTensor.shape == inputTensor.shape, (
		f"l2norm returned shape {tuple(resultTensor.shape)}, expected {tuple(inputTensor.shape)} "
		f"for input shape {tuple(t.shape)}."
	)

	vectorNorms = torch.sqrt((resultTensor * resultTensor).sum(dim=-1))
	expectedNorms = torch.ones_like(vectorNorms)
	assert torch.allclose(vectorNorms, expectedNorms, atol=tolerance), (
		f"l2norm produced non-unit vector norms {vectorNorms} for input shape {tuple(t.shape)} "
		f"with {tolerance=}."
	)

@pytest.mark.parametrize("eps", [pytest.param(1e-5, id="eps-default")])
def test_masked_mean(t: Tensor, boolean_mask_like_t: Tensor, reduction_dim: int | None, eps: float) -> None:
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
	("tensor_dtype", "tolerance"),
	[
		pytest.param(torch.float32, 1e-5, id="float32"),
		pytest.param(torch.float64, 1e-10, id="float64"),
	],
)
def test_RMSNorm(t: Tensor, tensor_dtype: torch.dtype, tolerance: float ) -> None:
	inputTensor = t.to(dtype=tensor_dtype)
	featureDimension = inputTensor.shape[-1]
	module = RMSNorm(featureDimension).to(dtype=tensor_dtype)

	resultTensor = module(inputTensor)
	expectedTensor = (
		torch.nn.functional.normalize(inputTensor, dim=-1)
		* (featureDimension ** 0.5)
		* module.gamma
	)
	resultRootMeanSquare = torch.sqrt(resultTensor.pow(2).mean(dim=-1))
	expectedRootMeanSquare = torch.ones_like(resultRootMeanSquare)

	assert tuple(module.gamma.shape) == (featureDimension,), (
		f"RMSNorm initialized gamma with shape {tuple(module.gamma.shape)}, expected {(featureDimension,)} "
		f"for input shape {tuple(inputTensor.shape)}."
	)
	assert resultTensor.shape == inputTensor.shape, (
		f"RMSNorm returned shape {tuple(resultTensor.shape)}, expected {tuple(inputTensor.shape)} "
		f"for {tensor_dtype=} and input shape {tuple(inputTensor.shape)}."
	)
	assert torch.allclose(resultTensor, expectedTensor, atol=tolerance, rtol=tolerance), (
		f"RMSNorm returned values {resultTensor} that do not match expected {expectedTensor} "
		f"for {tensor_dtype=}, input shape {tuple(inputTensor.shape)}, and {tolerance=}."
	)
	assert torch.allclose(resultRootMeanSquare, expectedRootMeanSquare, atol=tolerance, rtol=tolerance), (
		f"RMSNorm produced root-mean-square values {resultRootMeanSquare}, expected {expectedRootMeanSquare} "
		f"for {tensor_dtype=}, input shape {tuple(inputTensor.shape)}, and {tolerance=}."
	)


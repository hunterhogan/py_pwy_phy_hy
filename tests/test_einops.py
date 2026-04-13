from collections.abc import Sequence
from py_pwy_phy_hy.einops import pack_one, pack_with_inverse, unpack_one
from torch import Tensor
import pytest
import torch

@pytest.mark.parametrize(
    ("pattern", "tensor_shift"),
    [pytest.param("b *", 7.0, id="pattern-b-star-shift-seven")],
)
def test_pack_with_inverse_round_trip_for_tensor_and_sequence(
    pack_input: Tensor | list[Tensor],
    pattern: str,
    tensor_shift: float,
) -> None:
    if torch.is_tensor(pack_input):
        packed, inverse = pack_with_inverse(pack_input, pattern)
        round_trip = inverse(packed, None)
        shifted_round_trip = inverse(packed + tensor_shift, None)

        assert torch.is_tensor(round_trip), (
            "pack_with_inverse round-trip output is not a tensor for tensor input."
        )
        assert torch.equal(round_trip, pack_input), (
            f"pack_with_inverse round-trip mismatch for tensor input with {tuple(pack_input.shape)=} and {pattern=}."
        )

        expected_shifted = pack_input + tensor_shift
        assert torch.is_tensor(shifted_round_trip), (
            "pack_with_inverse shifted round-trip output is not a tensor for tensor input."
        )
        assert torch.equal(shifted_round_trip, expected_shifted), (
            f"pack_with_inverse shifted round-trip mismatch for tensor input with {tuple(pack_input.shape)=} and {pattern=}."
        )
        return

    assert isinstance(pack_input, list), (
        "pack_with_inverse list test input must be a list of tensors."
    )
    packed, inverse = pack_with_inverse(pack_input, pattern)
    round_trip = inverse(packed, None)
    shifted_round_trip = inverse(packed + tensor_shift, None)

    assert isinstance(round_trip, list), (
        "pack_with_inverse round-trip output is not a list for sequence input."
    )
    assert isinstance(shifted_round_trip, list), (
        "pack_with_inverse shifted output is not a list for sequence input."
    )
    assert len(round_trip) == len(pack_input), (
        f"pack_with_inverse round-trip sequence length mismatch: got {len(round_trip)}, expected {len(pack_input)}."
    )
    assert len(shifted_round_trip) == len(pack_input), (
        f"pack_with_inverse shifted sequence length mismatch: got {len(shifted_round_trip)}, expected {len(pack_input)}."
    )

    for tensor_index, (expected_tensor, round_trip_tensor, shifted_tensor) in enumerate(
        zip(pack_input, round_trip, shifted_round_trip, strict=True)
    ):
        assert torch.equal(round_trip_tensor, expected_tensor), (
            f"pack_with_inverse round-trip tensor mismatch at index {tensor_index} for {pattern=}."
        )
        assert torch.equal(shifted_tensor, expected_tensor + tensor_shift), (
            f"pack_with_inverse shifted tensor mismatch at index {tensor_index} for {pattern=}."
        )


def test_pack_one_returns_packed_tensor_with_shape_metadata(
    t: Tensor,
    einops_pack_one_pattern: str,
) -> None:
    packed, packedShape = pack_one(t, einops_pack_one_pattern)

    assert packed.numel() == t.numel(), (
        f"pack_one packed tensor has {packed.numel()} elements, expected {t.numel()} "
        f"for input shape {tuple(t.shape)} and {einops_pack_one_pattern=}."
    )
    assert len(packedShape) == 1, (
        f"pack_one packed_shape has {len(packedShape)} entries, expected 1 "
        f"for single-tensor input with shape {tuple(t.shape)}."
    )


def test_unpack_one_round_trip_restores_original_tensor(
    t: Tensor,
    pack_one_result: tuple[Tensor, Sequence[tuple[int, ...] | list[int]]],
    einops_pack_one_pattern: str,
) -> None:
    packed, packedShape = pack_one_result
    result = unpack_one(packed, packedShape, einops_pack_one_pattern)

    assert result.shape == t.shape, (
        f"unpack_one returned shape {tuple(result.shape)}, expected {tuple(t.shape)} "
        f"for {einops_pack_one_pattern=}."
    )
    assert torch.equal(result, t), (
        f"unpack_one round-trip mismatch for input shape {tuple(t.shape)} and {einops_pack_one_pattern=}."
    )

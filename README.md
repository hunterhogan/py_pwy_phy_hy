# py-pwy-phy-hy

Typed tensor-shaping, masking, padding, device-routing, and checkpoint utilities for PyTorch and `einops`.

This repository is a fork of [`lucidrains/torch-einops-utils`](https://github.com/lucidrains/torch-einops-utils). The upstream repository is a compact collection of small utilities that show up repeatedly in lucidrains model repositories. `py-pwy-phy-hy` keeps that role. The main difference is emphasis. This fork adds roughly 6000 lines of tests, typing, and docstrings so the utility layer is easier to trust, easier to search, and easier to apply correctly.

`py-pwy-phy-hy` is most useful when combined with other lucidrains repositories. Repositories such as [`dreamer4`](https://github.com/lucidrains/dreamer4), [`metacontroller`](https://github.com/lucidrains/metacontroller), [`mimic-video`](https://github.com/lucidrains/mimic-video), [`pi-zero-pytorch`](https://github.com/lucidrains/pi-zero-pytorch), [`sdft-pytorch`](https://github.com/lucidrains/sdft-pytorch), and [`locoformer`](https://github.com/lucidrains/locoformer) repeatedly need operations such as `align_dims_left`, `shape_with_replace`, `lens_to_mask`, `pad_sequence`, `safe_cat`, and `pack_with_inverse`. This package centralizes those operations in one typed import surface instead of re-implementing the same tensor utility layer in each model repository.

If you already know `torch-einops-utils`, the best mental model is simple. `py_pwy_phy_hy` is a typed substitute for that package. The public function family is the same kind of function family: small PyTorch and `einops` helpers for shape work, masks, padding, optional tensors, PyTree traversal, device routing, and checkpoint reconstruction. The important difference is import path. Upstream code imports `torch_einops_utils`. This repository exports `py_pwy_phy_hy`. The substitution is conceptual, not literal import-path compatibility.

Use `py-pwy-phy-hy` when you want strict typing, a `py.typed` marker, focused modules, extensive tests, and docstrings written for both humans and AI assistants. Use upstream when you want the most compact possible version of the same idea.

## At a glance

- Project name: `py-pwy-phy-hy`.
- Import path: `py_pwy_phy_hy`.
- Python requirement: `>=3.10`.
- Runtime dependencies: `torch`, `einops`, `packaging`, and `typing-extensions`.
- Root package exports: helper functions, slicing helpers, rank-alignment helpers, mask helpers, safe concatenation helpers, padding helpers, and PyTree / `einops` helpers.
- Submodules with dedicated imports: `py_pwy_phy_hy.device` and `py_pwy_phy_hy.save_load`.
- Typing status: the package ships a `py.typed` marker and the repository uses strict type checking.
- Best fit: lucidrains-style model repositories that work with variable-length tensors, `einops` patterns, optional intermediate tensors, and nested `torch.nn.Module` graphs.

## Installation

Install from this repository with `uv`:

```bash
uv add git+https://github.com/hunterhogan/py_pwy_phy_hy.git
```

## Import map

Import most tensor helpers from the package root:

```python
from py_pwy_phy_hy import (
    align_dims_left,
    and_masks,
    lens_to_mask,
    masked_mean,
    maybe,
    or_masks,
    pack_with_inverse,
    pad_sequence,
    pad_sequence_and_cat,
    safe_cat,
    safe_stack,
    shape_with_replace,
    slice_at_dim,
    tree_flatten_with_inverse,
    tree_map_tensor,
)
```

Import device decorators from `py_pwy_phy_hy.device`:

```python
from py_pwy_phy_hy.device import (
    module_device,
    move_inputs_to_device,
    move_inputs_to_module_device,
)
```

Import checkpoint decorators from `py_pwy_phy_hy.save_load`:

```python
from py_pwy_phy_hy.save_load import (
    dehydrate_config,
    rehydrate_config,
    save_load,
)
```

## Quick examples

### Batch variable-length tensors and build a mask

This example comes directly from the test suite.

```python
import torch

from py_pwy_phy_hy import lens_to_mask, pad_sequence

x = torch.randn(2, 4, 5)
y = torch.randn(2, 3, 5)
z = torch.randn(2, 1, 5)

packed, lens = pad_sequence([x, y, z], dim=1, return_lens=True)
mask = lens_to_mask(lens)

assert packed.shape == (3, 2, 4, 5)
assert lens.tolist() == [4, 3, 1]
assert torch.allclose(mask.sum(dim=-1), lens)
```

### Pack with `einops` and keep an inverse function

This example also comes directly from the test suite.

```python
import torch

from py_pwy_phy_hy import pack_with_inverse

t = torch.randn(3, 12, 2, 2)
packed, inverse = pack_with_inverse(t, "b * d")

assert packed.shape == (3, 24, 2)

restored = inverse(packed)
assert restored.shape == (3, 12, 2, 2)
```

### Decorate a module with save / load helpers

This example is adapted from `tests/test_save_load.py`.

```python
from pathlib import Path

from torch import Tensor, nn

from py_pwy_phy_hy.save_load import save_load


@save_load()
class SimpleNet(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.net = nn.Linear(dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


path = Path("model.pt")
model = SimpleNet(10, 20)
model.save(str(path))

restored = SimpleNet.init_and_load(str(path))
assert restored.dim == 10
assert restored.hidden_dim == 20
```

## Root API reference

### Optional-value and structure helpers

| Name                           | Contract                                                                                                                                                                                                                      |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `exists(v)`                    | Returns `True` exactly when `v is not None`. Falsy values such as `0`, `False`, and `[]` still count as existing values. The return type is a `TypeGuard`, so static analyzers narrow the type after the check.               |
| `default(v, d)`                | Returns `v` when `v` exists. Returns `d` when `v is None`.                                                                                                                                                                    |
| `compact(arr)`                 | Removes every `None` value from `arr` and returns a `list` of the remaining values.                                                                                                                                           |
| `divisible_by(num, den)`       | Returns `False` when `den == 0`. Otherwise returns whether `num % den == 0`.                                                                                                                                                  |
| `identity(t, *args, **kwargs)` | Returns `t` unchanged and ignores every extra argument. Useful as a no-op callable.                                                                                                                                           |
| `first(arr)`                   | Returns `arr[0]`. Use `first` when the sequence supports integer indexing and index `0` is valid.                                                                                                                             |
| `map_values(fn, v)`            | Recursively applies `fn` to every leaf value inside nested `list`, `tuple`, and `dict` structures. Container shape is preserved.                                                                                              |
| `maybe(fn)`                    | Wraps `fn` so the wrapped callable returns `None` when the first argument is `None`. When `fn` itself is `None`, `maybe(None)` returns `identity`.                                                                            |
| `safe(fn)`                     | Decorator for callables whose first argument is `Sequence[Tensor]`. The decorator removes `None` values from that first argument before the call. If no tensors remain, the decorator returns `None` instead of calling `fn`. |

### Shape and slicing helpers

| Name                                       | Contract                                                                                                                                                      |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `shape_with_replace(t, replace_dict=None)` | Returns a derived `torch.Size` based on `t.shape`. The function does not modify `t`. Keys in `replace_dict` must be non-negative integers less than `t.ndim`. |
| `slice_at_dim(t, slc, dim=-1)`             | Applies `slc` to one dimension and preserves every other dimension. Negative `dim` values are normalized before indexing.                                     |
| `slice_left_at_dim(t, length, dim=-1)`     | Keeps the first `length` values along `dim`. When `length == 0`, the function returns an empty slice along `dim`.                                             |
| `slice_right_at_dim(t, length, dim=-1)`    | Keeps the last `length` values along `dim`. When `length == 0`, the function returns an empty slice along `dim`.                                              |

### Rank-alignment and singleton-dimension helpers

These functions change shape by inserting singleton dimensions. These functions do not add numeric padding values.

| Name                                  | Contract                                                                                                                                                                                                   |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pad_ndim(t, (left, right))`          | Returns a reshaped view of `t` with `left` leading singleton dimensions and `right` trailing singleton dimensions. Raises `ValueError` when either count is negative.                                      |
| `pad_left_ndim(t, ndims)`             | Prepends `ndims` singleton dimensions.                                                                                                                                                                     |
| `pad_right_ndim(t, ndims)`            | Appends `ndims` singleton dimensions.                                                                                                                                                                      |
| `pad_left_ndim_to(t, ndims)`          | Ensures that `t.ndim >= ndims` by prepending singleton dimensions when needed. Returns `t` unchanged when `t.ndim >= ndims`.                                                                               |
| `pad_right_ndim_to(t, ndims)`         | Ensures that `t.ndim >= ndims` by appending singleton dimensions when needed. Returns `t` unchanged when `t.ndim >= ndims`.                                                                                |
| `align_dims_left(tensors, ndim=None)` | Aligns every tensor to a shared rank by appending trailing singleton dimensions. Existing dimensions stay left-aligned. When `ndim is None`, the target rank is the maximum rank across the input tensors. |

### Mask helpers

| Name                               | Contract                                                                                                                                             |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lens_to_mask(lens, max_len=None)` | Converts integer length values to a boolean mask with shape `(*lens.shape, max_len)`. Position `i` in the final axis is `True` when `i < lens[...]`. |
| `reduce_masks(masks, op)`          | Filters out `None` values, then reduces the remaining masks left-to-right with `op`. Returns `None` when no non-`None` masks remain.                 |
| `and_masks(masks)`                 | Equivalent to `reduce_masks(masks, torch.logical_and)`. Returns `None` when no active mask remains.                                                  |
| `or_masks(masks)`                  | Equivalent to `reduce_masks(masks, torch.logical_or)`. Returns `None` when no active mask remains.                                                   |

### Safe concatenation and stacking helpers

| Name                         | Contract                                                                                                                                                                                        |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `safe_stack(tensors, dim=0)` | Removes `None` values, then applies `torch.stack`. Returns `None` when no active tensor remains. Even a single tensor is stacked, so the result gains a new dimension.                          |
| `safe_cat(tensors, dim=0)`   | Removes `None` values, then applies `torch.cat`. Returns `None` when no active tensor remains. A single surviving tensor is returned unchanged because `torch.cat` over one tensor is identity. |

### Numeric padding and batching helpers

These functions add numeric padding values along an existing tensor dimension.

| Name                                                                      | Contract                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pad_at_dim(t, (left, right), dim=-1, value=0.0)`                         | Pads or trims `t` along one dimension. Positive values add elements. Negative values trim elements.                                                                                                                                      |
| `pad_left_at_dim(t, pad, dim=-1, value=0.0)`                              | Prepends `pad` values along `dim`.                                                                                                                                                                                                       |
| `pad_right_at_dim(t, pad, dim=-1, value=0.0)`                             | Appends `pad` values along `dim`.                                                                                                                                                                                                        |
| `pad_left_at_dim_to(t, length, dim=-1, value=0.0)`                        | Ensures that `t.shape[dim] >= length` by left-padding when needed. Returns `t` unchanged when the target length is already satisfied.                                                                                                    |
| `pad_right_at_dim_to(t, length, dim=-1, value=0.0)`                       | Ensures that `t.shape[dim] >= length` by right-padding when needed. Returns `t` unchanged when the target length is already satisfied.                                                                                                   |
| `pad_sequence(tensors, ...)`                                              | Pads every tensor in `tensors` to the maximum length along `dim`. The function can return a stacked tensor or a list of padded tensors. The function can also return original lengths or padding widths. Returns `None` for empty input. |
| `pad_sequence_and_cat(tensors, dim_cat=0, dim=-1, value=0.0, left=False)` | Equivalent to `pad_sequence(..., return_stacked=False)` followed by `torch.cat(..., dim=dim_cat)`. Returns `None` for empty input.                                                                                                       |

#### `pad_sequence` return modes

`pad_sequence` is overloaded. The return type depends on `return_stacked` and `return_lens`.

| `return_stacked` | `return_lens` | Return type                           |
| ---------------- | ------------- | ------------------------------------- |
| `True`           | `False`       | `Tensor \| None`                      |
| `True`           | `True`        | `tuple[Tensor, Tensor] \| None`       |
| `False`          | `False`       | `list[Tensor] \| None`                |
| `False`          | `True`        | `tuple[list[Tensor], Tensor] \| None` |

When `pad_lens=True` and `return_lens=True`, the second tensor contains padding widths rather than original lengths.

### PyTree and `einops` helpers

| Name                                            | Contract                                                                                                                                                                                                                                                                                       |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `masked_mean(t, mask=None, dim=None, eps=1e-5)` | Computes a masked mean. When `mask is None`, the function falls back to `t.mean(...)`. When no masked position is selected and `dim is None`, the function returns zero by summing over the empty selection. When `mask.ndim < t.ndim`, the function right-pads mask rank before broadcasting. |
| `tree_map_tensor(fn, tree)`                     | Applies `fn` to every tensor leaf in a PyTree and leaves non-tensor leaves unchanged.                                                                                                                                                                                                          |
| `tree_flatten_with_inverse(tree)`               | Returns a flat list of leaves and an inverse function that reconstructs the original PyTree shape from a replacement iterable of leaves.                                                                                                                                                       |
| `pack_with_inverse(t, pattern)`                 | Wraps `einops.pack` and stores the corresponding inverse. Input `t` may be one tensor or a `list[Tensor]`. The inverse returns the matching kind and optionally accepts a different `inv_pattern` for unpacking.                                                                               |

## `device` submodule reference

The `py_pwy_phy_hy.device` submodule contains three utilities for device inference and automatic tensor movement.

| Name                               | Contract                                                                                                                                                                                                                                                        |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `module_device(m)`                 | Returns the device of the first parameter or registered buffer in `m`. Returns `None` when `m` has neither parameters nor buffers.                                                                                                                              |
| `move_inputs_to_device(device)`    | Decorator that recursively moves every tensor inside positional and keyword arguments to `device` before calling the wrapped function. Non-tensor values pass through unchanged.                                                                                |
| `move_inputs_to_module_device(fn)` | Decorator for methods whose first argument is a `torch.nn.Module`. The decorator infers the target device with `module_device(self)` and moves every tensor argument after `self` to that device. If `module_device(self)` returns `None`, the call is a no-op. |

## `save_load` submodule reference

The `py_pwy_phy_hy.save_load` submodule contains the checkpoint decorator and the two advanced configuration helpers that support nested decorated modules.

| Name                                                 | Contract                                                                                                                                                                                                                    |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `save_load(...)`                                     | Class decorator for `torch.nn.Module` subclasses. The decorator records constructor arguments on each instance and adds instance save / load methods plus a classmethod that reconstructs a new instance from a checkpoint. |
| `dehydrate_config(config, config_instance_var_name)` | Walks nested `list`, `tuple`, and `dict` structures and replaces decorated module instances with reconstruction records.                                                                                                    |
| `rehydrate_config(config)`                           | Walks nested `list`, `tuple`, and `dict` structures and reconstructs decorated modules from the stored reconstruction records.                                                                                              |

### Default `save_load` method names

The default decorator call is `@save_load()`. That default adds:

- `save(path, overwrite=True)`
- `load(path, strict=True)`
- `init_and_load(path, strict=True)`

The decorator can rename all three methods and can rename the instance attribute that stores constructor configuration.

### Checkpoint contents

The generated checkpoint payload stores:

- `model`: the result of `state_dict()`.
- `config`: a pickled constructor-configuration payload.
- `version`: an optional version string.

### Nested module reconstruction

Nested module graphs are a first-class use case. If one decorated module is passed as a constructor argument to another decorated module, `save_load` can dehydrate that nested module graph during save and rehydrate that nested module graph during `init_and_load`.

### Version behavior

When both the checkpoint version and the current decorator version exist and differ, `load` emits `UserWarning` and still loads the checkpoint state.

## Edge-case conventions

These conventions matter for both humans and AI assistants.

- `exists` tests only `None`. `0`, `False`, empty strings, empty lists, and empty dictionaries still count as existing values.
- `maybe(fn)` short-circuits on the first argument only. `maybe(fn)(None, ...)` returns `None` without calling `fn`.
- `safe`, `safe_cat`, `safe_stack`, `reduce_masks`, `and_masks`, and `or_masks` all treat `None` values as absent values rather than false values.
- `safe_cat`, `safe_stack`, `reduce_masks`, `and_masks`, `or_masks`, `pad_sequence`, and `pad_sequence_and_cat` can return `None` for empty effective input. Generated code must account for that return path.
- `pad_left_at_dim_to`, `pad_right_at_dim_to`, `pad_left_ndim_to`, and `pad_right_ndim_to` return the original tensor unchanged when no growth is needed.
- `shape_with_replace` accepts only non-negative dimension keys. `slice_at_dim`, `slice_left_at_dim`, and `slice_right_at_dim` do accept negative `dim` values.
- `module_device` returns `None` for stateless modules such as `nn.Identity()`.
- `move_inputs_to_module_device` becomes a no-op when `module_device(self)` returns `None`.
- `save_load.load` may warn on version mismatch but still restore model state.
- `save_load.init_and_load` requires a checkpoint that contains serialized constructor configuration.

## Typing and AI-assistant notes

This repository is written to be readable by humans, machine translation systems, search tools, and AI assistants.

- The package ships a `py.typed` marker.
- The repository uses strict type checking in `pyrightconfig.json`.
- `exists` uses `TypeGuard`, so a type checker can narrow `T | None` to `T` after an `exists(...)` check.
- `pad_sequence`, `pack_with_inverse`, and `maybe` use overloads so type checkers can preserve useful return information.
- The documentation keeps identifier names stable. `pad_sequence` always means `pad_sequence`. `safe_cat` always means `safe_cat`. This consistency helps retrieval and generated code.

If you are an AI assistant adapting code from `torch-einops-utils`, use these translation rules:

- Replace `from torch_einops_utils import ...` with `from py_pwy_phy_hy import ...` for root helpers.
- Replace upstream `save_load` imports with `from py_pwy_phy_hy.save_load import save_load`.
- Replace upstream device decorator imports with `from py_pwy_phy_hy.device import ...`.
- Do not erase `None`-return paths with unconditional casts. Several helpers intentionally return `None` for empty effective input.
- Do not assume `pad_sequence` always returns a stacked tensor. Read `return_stacked` and `return_lens` first.

## Differences from upstream `torch-einops-utils`

This repository is not a repackaged mirror of upstream. This repository makes a different trade-off.

- Upstream is intentionally compact.
- This fork splits the implementation across focused modules such as `_helpers.py`, `_padding.py`, `device.py`, and `save_load.py` while still re-exporting most tensor helpers from the package root.
- This fork adds strict typing, a `py.typed` marker, extensive tests, and detailed docstrings.
- This fork is best treated as a typed, documented branch of the same utility idea rather than a literal import-path-compatible drop-in replacement.

## Repository layout

- `src/py_pwy_phy_hy/` — package source.
- `src/py_pwy_phy_hy/device.py` — device inference and input-routing decorators.
- `src/py_pwy_phy_hy/save_load.py` — checkpoint save / load decorator and nested reconstruction helpers.
- `tests/` — regression tests and usage examples for helpers, masks, padding, device routing, and checkpoint reconstruction.

## Development

Set up the repository:

```bash
uv sync
```

Run the test suite:

```bash
pytest
```

Run static analysis:

```bash
pyright
ruff check .
```

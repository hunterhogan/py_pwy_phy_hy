"""Determine the compute device of a `torch.nn.Module` instance and route `Tensor` arguments to that device or an explicit target `Device`.

You can use this Python module to determine the compute device on which a `torch.nn.Module` instance
resides and to decorate callables so that `Tensor` arguments are automatically moved to the
appropriate compute device before each call. These utilities remove explicit `.to(device)` calls from
each call site and centralize device routing in a single Python decorator.

Contents
--------
Functions
	module_device
		Infer the `torch.device` of a `torch.nn.Module` instance from its first `torch.nn.Parameter`
		or registered `torch.Tensor` buffer.
	move_inputs_to_device
		Create a Python decorator that moves all `Tensor` arguments to a fixed target compute device.
	move_inputs_to_module_device
		Wrap a callable so that all `Tensor` arguments are moved to the `torch.device` of the first
		`torch.nn.Module` argument before each call.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from itertools import chain
from py_pwy_phy_hy import exists, PSpec, TorchNNModule, tree_map_tensor, TVar
from torch import device, Tensor
from torch.nn import Module
from torch.types import Device
from typing import Concatenate

def module_device(m: Module) -> device | None:
	"""Infer the `torch.device` of a `torch.nn.Module` instance from its first `torch.nn.Parameter` or registered `torch.Tensor` buffer.

	You can use `module_device` to determine the compute device of a `torch.nn.Module` instance
	without reading the instance's internals directly. `module_device` checks the learnable
	`torch.nn.Parameter` values of `m` before checking the registered `torch.Tensor` buffer values of
	`m`, and returns the `torch.device` of the first tensor found. If `m` has no learnable
	`torch.nn.Parameter` values and no registered `torch.Tensor` buffer values, `module_device`
	returns `None`.

	PyTorch Details
	---------------
	A `torch.nn.Module` instance tracks two kinds of tensors that reside on a specific compute
	device: learnable `torch.nn.Parameter` values, returned by `Module.parameters` [2], and
	registered `torch.Tensor` buffer values, returned by `Module.buffers` [3]. A registered
	`torch.Tensor` buffer is a named tensor registered via `Module.register_buffer` [4] that
	participates in device movement through `.to()` and `.cuda()` but is not updated by
	gradient-based optimizers. `module_device` passes the return of `Module.parameters` and the
	return of `Module.buffers` to `itertools.chain` [5], takes the first element, and returns that
	tensor's `.device` attribute.

	Parameters
	----------
	m : Module
		The `torch.nn.Module` instance whose compute device is to be inferred.

	Returns
	-------
	inferredDevice : torch.device | None
		The `torch.device` of the first `torch.nn.Parameter` or registered `torch.Tensor` buffer in
		`m`, or `None` if `m` has no `torch.nn.Parameter` values and no registered `torch.Tensor`
		buffer values.

	See Also
	--------
	move_inputs_to_module_device : Wrap a callable so that all `Tensor` arguments are moved to the
		`torch.device` of the first `torch.nn.Module` argument before each call.

	Examples
	--------
	Retrieve the `torch.device` of a `torch.nn.Module` instance with `torch.nn.Parameter` values [6]:

		```python
		import torch
		from torch import nn
		from py_pwy_phy_hy.device import module_device

		linear = nn.Linear(3, 5)
		inferredDevice = module_device(linear)
		# inferredDevice == torch.device('cpu')
		```

	Returns `None` for a `torch.nn.Module` instance that has no `torch.nn.Parameter` values and no
	registered `torch.Tensor` buffer values:

		```python
		empty = nn.Identity()
		inferredDevice = module_device(empty)
		# inferredDevice is None
		```

	References
	----------
	[1] torch.nn.Module - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.nn.Module.html
	[2] torch.nn.Module.parameters - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters
	[3] torch.nn.Module.buffers - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.buffers
	[4] torch.nn.Module.register_buffer - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
	[5] itertools.chain - Python documentation
		https://docs.python.org/3/library/itertools.html#itertools.chain
	[6] tests.test_device_part2.test_module_device_returns_expected_device

	"""
	first_param_or_buffer: Tensor | None = next(chain(m.parameters(), m.buffers()), None)

	if not exists(first_param_or_buffer):
		return None

	return first_param_or_buffer.device

def move_inputs_to_device(device: Device) -> Callable[[Callable[PSpec, TVar]], Callable[PSpec, TVar]]:
	"""Create a Python decorator that moves all `Tensor` arguments to a target compute device before each call.

	You can use `move_inputs_to_device` to wrap a callable so that every `Tensor` in its positional
	and keyword arguments is moved to the compute device identified by `device` before the wrapped
	callable is called. Non-`Tensor` arguments pass through without modification.
	`move_inputs_to_device` uses `tree_map_tensor` [2] to recurse into containers such as `tuple` and
	`dict`, so `Tensor` values nested inside those containers are moved as well.

	PyTorch Details
	---------------
	The `device` argument accepts any `Device` value [3], including a `str` such as `"cpu"` or
	`"cuda"`, an `int` CUDA device index, a `torch.device` instance, or `None`. `tree_map_tensor`
	applies `Tensor.to(device)` [4] to every `Tensor` found recursively in `args` and `kwargs`. The
	wrapped callable's signature is preserved by `functools.wraps`.

	Parameters
	----------
	device : Device
		The target compute device to which all `Tensor` arguments are moved. Accepts any value valid
		for `Tensor.to` [4], such as `"cpu"`, `"cuda"`, an integer CUDA device index, or a
		`torch.device` instance.

	Returns
	-------
	decorator : Callable[[Callable[PSpec, TVar]], Callable[PSpec, TVar]]
		A Python decorator that wraps the given callable, moving its `Tensor` arguments to the
		compute device identified by `device` before each call while preserving the original
		callable's signature.

	See Also
	--------
	move_inputs_to_module_device : Infer the target compute device from a `torch.nn.Module` instance
		rather than accepting an explicit `Device` argument.

	Examples
	--------
	Wrap a callable so all `Tensor` arguments are moved to the `meta` compute device [5]:

		```python
		import torch
		from py_pwy_phy_hy.device import move_inputs_to_device

		targetDevice = torch.device("meta")

		@move_inputs_to_device(targetDevice)
		def collectDeviceTypes(
			positionTensor: torch.Tensor,
			nestedTuple: tuple[torch.Tensor, str],
			*,
			keywordTensor: torch.Tensor,
		) -> tuple[torch.device, torch.device, torch.device]:
			return positionTensor.device, nestedTuple[0].device, keywordTensor.device

		cpuTensor = torch.tensor([1.0, 2.0])
		result = collectDeviceTypes(cpuTensor, (cpuTensor, "north"), keywordTensor=cpuTensor)
		# result[0] == torch.device("meta")
		# result[1] == torch.device("meta")
		# result[2] == torch.device("meta")
		```

	References
	----------
	[1] torch.Tensor - PyTorch documentation
		https://pytorch.org/docs/stable/tensors.html
	[2] py_pwy_phy_hy.tree_map_tensor

	[3] torch.types.Device - PyTorch documentation
		https://pytorch.org/docs/stable/tensor_attributes.html
	[4] torch.Tensor.to - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
	[5] tests.test_device_part2.test_move_inputs_to_device_moves_tensor_arguments_in_nested_structures

	"""

	def decorator(fn: Callable[PSpec, TVar]) -> Callable[PSpec, TVar]:
		@wraps(fn)
		def inner(*args: PSpec.args, **kwargs: PSpec.kwargs) -> TVar:
			args, kwargs = tree_map_tensor(lambda t: t.to(device), (args, kwargs))

			return fn(*args, **kwargs)

		return inner

	return decorator

def move_inputs_to_module_device(fn: Callable[Concatenate[TorchNNModule, PSpec], TVar]) -> Callable[Concatenate[TorchNNModule, PSpec], TVar]:
	"""Wrap a callable so that all `Tensor` arguments are moved to the `torch.device` of the first `torch.nn.Module` argument before each call.

	You can use `move_inputs_to_module_device` as a Python decorator on methods of `torch.nn.Module`
	subclasses [1] to automatically move all `Tensor` arguments to the compute device of the
	`torch.nn.Module` instance. `move_inputs_to_module_device` inspects the first argument (`self`)
	using `module_device` [2] to determine the target `torch.device`, then uses `tree_map_tensor` [3]
	to move every `Tensor` found in the remaining positional and keyword arguments. If the
	`torch.nn.Module` instance has no `torch.nn.Parameter` values and no registered `torch.Tensor`
	buffer values, the `Tensor` arguments pass through without modification.

	The first argument of the decorated callable must be a `torch.nn.Module` instance.

	PyTorch Details
	---------------
	`typing.Concatenate` [4] expresses the constraint that the first argument of `fn` is a
	`torch.nn.Module` instance while the remaining arguments form `PSpec`. The `TypeVar` `TorchNNModule`
	is bound to `Module`, so the type of the first argument reflects the specific `torch.nn.Module`
	subclass. `tree_map_tensor` applies `Tensor.to(device)` to every `Tensor` found recursively in
	the non-`self` arguments. The wrapped callable's signature is preserved by `functools.wraps`.

	Parameters
	----------
	fn : Callable[Concatenate[TorchNNModule, PSpec], TVar]
		The callable to wrap. The first positional argument must be a `torch.nn.Module` instance
		whose `torch.device` is used as the routing target.

	Returns
	-------
	wrappedMethod : Callable[Concatenate[TorchNNModule, PSpec], TVar]
		The wrapped callable with the same signature as `fn`, where all `Tensor` arguments after the
		first `torch.nn.Module` argument are moved to the `torch.device` of that instance before each
		call.

	See Also
	--------
	module_device : Infer the `torch.device` of a `torch.nn.Module` instance from its first
		`torch.nn.Parameter` or registered `torch.Tensor` buffer.
	move_inputs_to_device : Move `Tensor` arguments to an explicit target compute device rather than
		inferring it from a `torch.nn.Module` instance.

	Examples
	--------
	Decorate a method so all `Tensor` arguments are moved to the `torch.device` of the
	`torch.nn.Module` instance [5]:

		```python
		import torch
		from torch import nn, Tensor
		from py_pwy_phy_hy.device import move_inputs_to_module_device

		class EchoModule(nn.Module):
			def __init__(self) -> None:
				super().__init__()
				self.scale = nn.Parameter(torch.tensor([2.0], device=torch.device("meta")))

			@move_inputs_to_module_device
			def forward(self, tensorValue: Tensor) -> Tensor:
				return tensorValue

		module = EchoModule()
		cpuTensor = torch.tensor([1.0, 2.0])
		result = module.forward(cpuTensor)
		# result.device == torch.device("meta")
		```

	Decorate a standalone callable and attach it to a `torch.nn.Module` subclass as a method [6]:

		```python
		@move_inputs_to_module_device
		def policy_loss(model, state, old_log_probs, actions, advantages, mask=None): ...

		MetaController.policy_loss = policy_loss
		```

	References
	----------
	[1] torch.nn.Module - PyTorch documentation
		https://pytorch.org/docs/stable/generated/torch.nn.Module.html
	[2] py_pwy_phy_hy.device.module_device

	[3] py_pwy_phy_hy.tree_map_tensor

	[4] typing.Concatenate - Python documentation
		https://docs.python.org/3/library/typing.html#typing.Concatenate
	[5] tests.test_device_part2._echo_module_with_parameter

	[6] metacontroller.metacontroller.policy_loss
		https://context7.com/lucidrains/metacontroller
	"""

	@wraps(fn)
	def inner(self: TorchNNModule, *args: PSpec.args, **kwargs: PSpec.kwargs) -> TVar:
		theDevice: device | None = module_device(self)

		if exists(theDevice):
			args, kwargs = tree_map_tensor(lambda t: t.to(theDevice), (args, kwargs))

		return fn(self, *args, **kwargs)  # ty:ignore[invalid-argument-type]

	return inner

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

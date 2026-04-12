from __future__ import annotations

from pathlib import Path
from py_pwy_phy_hy.save_load import save_load
from torch import nn, Tensor
import torch

@save_load()
class SimpleNet(nn.Module):
    def __init__(self: SimpleNet, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dim: int = dim
        self.hidden_dim: int = hidden_dim
        self.net: nn.Linear = nn.Linear(dim, hidden_dim)

    def forward(self: SimpleNet, x: Tensor) -> Tensor:
        return self.net(x)

def test_save_load() -> None:
    model: SimpleNet = SimpleNet(10, 20)
    path: Path = Path('test_model.pt')

    # Save the model
    model.save(str(path))

    # Create another model with different weights
    model2: SimpleNet = SimpleNet(10, 20)

    # Ensure weights are different initially
    assert not torch.allclose(model.net.weight, model2.net.weight)

    # Load back
    model2.load(str(path))

    # Validate weights are the same
    assert torch.allclose(model.net.weight, model2.net.weight)

    # Cleanup
    if path.exists():
        path.unlink()

def test_init_and_load() -> None:
    dim: int = 16
    hidden_dim: int = 32
    model: SimpleNet = SimpleNet(dim, hidden_dim)
    path: Path = Path('test_model_init.pt')

    # Save the model
    model.save(str(path))

    # Initialize and load from file
    model2: SimpleNet = SimpleNet.init_and_load(str(path))

    # Validate attributes and weights
    assert model2.dim == dim
    assert model2.hidden_dim == hidden_dim
    assert torch.allclose(model.net.weight, model2.net.weight)

    # Cleanup
    if path.exists():
        path.unlink()

# nested save load

@save_load()
class GrandChild(nn.Module):
    def __init__(self: GrandChild, dim: int) -> None:
        super().__init__()
        self.dim: int = dim
        self.param: nn.Parameter = nn.Parameter(torch.randn(dim))

@save_load()
class Child(nn.Module):
    def __init__(self: Child, grandchild: GrandChild | None = None, name: str = 'child') -> None:
        super().__init__()
        self.grandchild: GrandChild | None = grandchild
        self.name: str = name
        self.param: nn.Parameter = nn.Parameter(torch.randn(1))

@save_load()
class Parent(nn.Module):
    def __init__(self: Parent, child1: Child, child2: Child | None = None) -> None:
        super().__init__()
        self.child1: Child = child1
        self.child2: Child | None = child2
        self.param: nn.Parameter = nn.Parameter(torch.randn(1))

@save_load()
class GrandParent(nn.Module):
    def __init__(self: GrandParent, p1: Parent, p2: Parent) -> None:
        super().__init__()
        self.p1: Parent = p1
        self.p2: Parent = p2
        self.param: nn.Parameter = nn.Parameter(torch.randn(1))

def test_sophisticated_nested_save_load() -> None:
    gc: GrandChild = GrandChild(dim = 8)
    c1: Child = Child(name = 'c1')
    c2: Child = Child(name = 'c2')
    c_nest: Child = Child(grandchild = gc, name = 'c_nest')

    p1: Parent = Parent(child1 = c1, child2 = c2)
    p2: Parent = Parent(child1 = c_nest)

    gp: GrandParent = GrandParent(p1 = p1, p2 = p2)

    path: Path = Path('sophisticated_test.pt')

    # Save
    gp.save(str(path))

    # Load
    gp2: GrandParent = GrandParent.init_and_load(str(path))

    # Verify structure
    assert gp2.p1.child1.name == 'c1'
    assert gp2.p1.child2 is not None
    assert gp2.p1.child2.name == 'c2'
    assert gp2.p2.child1.name == 'c_nest'
    assert gp2.p2.child1.grandchild is not None
    assert gp2.p2.child1.grandchild.dim == 8

    # Verify weight parity
    assert torch.allclose(gp.param, gp2.param)
    assert torch.allclose(gp.p1.param, gp2.p1.param)
    assert torch.allclose(gp.p1.child1.param, gp2.p1.child1.param)
    assert gp.p2.child1.grandchild is not None
    assert gp2.p2.child1.grandchild is not None
    assert torch.allclose(gp.p2.child1.grandchild.param, gp2.p2.child1.grandchild.param)

    if path.exists():
        path.unlink()

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

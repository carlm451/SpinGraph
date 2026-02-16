"""Lattice registry mapping names to generator classes.

Provides a single lookup point for all available lattice types in the zoo.
"""
from .square import SquareGenerator
from .kagome import KagomeGenerator
from .shakti import ShaktiGenerator
from .tetris import TetrisGenerator
from .santa_fe import SantaFeGenerator

LATTICE_REGISTRY = {
    'square': SquareGenerator,
    'kagome': KagomeGenerator,
    'shakti': ShaktiGenerator,
    'tetris': TetrisGenerator,
    'santa_fe': SantaFeGenerator,
}


def get_generator(name: str):
    """Get a lattice generator by name.

    Args:
        name: Lattice name (e.g., 'square', 'shakti', 'kagome').

    Returns:
        An instance of the corresponding LatticeGenerator subclass.

    Raises:
        KeyError: If the lattice name is not in the registry.
    """
    if name not in LATTICE_REGISTRY:
        available = ', '.join(sorted(LATTICE_REGISTRY.keys()))
        raise KeyError(
            f"Unknown lattice '{name}'. Available: {available}"
        )
    return LATTICE_REGISTRY[name]()


def list_lattices():
    """Return sorted list of available lattice names."""
    return sorted(LATTICE_REGISTRY.keys())

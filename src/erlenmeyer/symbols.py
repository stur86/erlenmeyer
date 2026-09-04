import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

Stochiometry = list[tuple[str, int]]


def _is_integer(value: Any) -> bool:
    """Tell if a value is an integer. Booleans are integers in Python, but not here."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_amount(amount: Any) -> bool:
    """Tell if a value can be used as a stochiometric amount."""
    return _is_integer(amount) and amount > 0


def _merge_stochiometry(terms: Stochiometry) -> Stochiometry:
    """Add together the amounts of repeated species, then sort by species."""
    merged: dict[str, int] = {}
    for species, amount in terms:
        merged[species] = merged.get(species, 0) + amount
    return sorted(merged.items())


def _format_stochiometry(terms: Stochiometry) -> str:
    """Write a stochiometry as a chemical sum, e.g. [("H", 2), ("O", 1)] -> "2H + O"."""
    if not terms:
        return "0"
    return " + ".join(
        f"{'' if amount == 1 else amount}{species}" for species, amount in terms
    )


class AbstractReactionTerm(ABC):
    @abstractmethod
    def stochiometry(self) -> Stochiometry: ...

    def __add__(self, other: object) -> "AbstractReactionTerm":
        if not isinstance(other, AbstractReactionTerm):
            raise TypeError(
                f"Unsupported sum between {type(self).__name__} and {type(other)}"
            )
        return _make_term(
            _merge_stochiometry(self.stochiometry() + other.stochiometry())
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AbstractReactionTerm):
            return self.stochiometry() == other.stochiometry()
        raise TypeError(
            f"Unsupported equality between {type(self).__name__} and {type(other)}"
        )

    def __hash__(self) -> int:
        return hash(tuple(self.stochiometry()))

    def __repr__(self) -> str:
        return _format_stochiometry(self.stochiometry())


class SingleReactionTerm(AbstractReactionTerm):
    _species: str
    _amount: int = 1

    _species_re = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")

    def __init__(self, species: str, amount: int = 1) -> None:
        assert isinstance(species, str), "Species must be a string"
        assert _is_amount(amount), "Amount must be an integer > 0"
        assert SingleReactionTerm._species_re.match(species), (
            "Species must start with a letter, then letters, numbers or underscores"
        )
        self._species = species
        self._amount = amount

    def stochiometry(self) -> Stochiometry:
        return [(self._species, self._amount)]

    @property
    def species(self):
        return self._species

    @property
    def amount(self):
        return self._amount


class Species(SingleReactionTerm):
    def __init__(self, species: str) -> None:
        super().__init__(species, 1)

    def __mul__(self, other: Any) -> SingleReactionTerm:
        if not _is_integer(other):
            raise TypeError(f"Can not multiply Species by {type(other)}")
        return SingleReactionTerm(species=self._species, amount=other)

    def __rmul__(self, other: Any) -> SingleReactionTerm:
        return self.__mul__(other)


class ReactionTerm(AbstractReactionTerm):
    _terms: Stochiometry

    def __init__(self, terms: Sequence[SingleReactionTerm | tuple[str, int]]) -> None:
        pairs: Stochiometry = []
        for t in terms:
            if not isinstance(t, SingleReactionTerm):
                t = SingleReactionTerm(*t)
            pairs.append(t.stochiometry()[0])
        self._terms = _merge_stochiometry(pairs)

    def stochiometry(self) -> Stochiometry:
        return self._terms


def _make_term(terms: Stochiometry) -> AbstractReactionTerm:
    """Build the simplest term that can hold a merged stochiometry."""
    if len(terms) == 1:
        return SingleReactionTerm(*terms[0])
    return ReactionTerm(terms)

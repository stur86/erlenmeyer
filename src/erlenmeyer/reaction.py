from dataclasses import dataclass

import numpy as np

from erlenmeyer.symbols import AbstractReactionTerm, Species


@dataclass(frozen=True)
class Reaction:
    """A single chemical reaction with a rate constant.

    The reaction is written ``reagents => products`` (or ``<=>`` when
    bidirectional). The rate is given as a float.
    """

    reagents: AbstractReactionTerm
    products: AbstractReactionTerm
    rate: float
    bidirectional: bool = False

    def __repr__(self) -> str:
        arrow = "<=>" if self.bidirectional else "=>"
        return f"{self.reagents} {arrow} {self.products} [{self.rate}]"


@dataclass(frozen=True)
class ReactionMatrices:
    """The stochiometric matrices of a set of reactions.

    Every reaction occupies one row in each matrix. ``reagents_m[i]`` and
    ``products_m[i]`` hold the amounts of each species consumed and produced by
    reaction ``i``, and ``rates_v[i]`` its rate constant.
    """

    reagents_m: np.ndarray
    products_m: np.ndarray
    rates_v: np.ndarray


class ReactionSystem:
    """A collection of species and the reactions that take place among them."""

    _species: list[Species]
    _reactions: set[Reaction]

    def __init__(self, species: list[Species]) -> None:
        self._species = []
        # Check that none of these is present twice
        for s in species:
            if s in self._species:
                raise ValueError(f"Species {s} included twice")
            self._species.append(s)
        self._reactions = set()

    def add_reaction(self, reaction: Reaction) -> None:
        """Add a reaction to the system.

        Raises ValueError if the reaction is already present.
        """
        if reaction in self._reactions:
            raise ValueError(f"Reaction {reaction} already present")
        self._reactions.add(reaction)

    def get_species_index(self, species: Species | str) -> int:
        """Return the index of a species in the system's species list.

        The species may be given either as a :class:`Species` or as its name.
        """
        if isinstance(species, str):
            species = Species(species)
        if species not in self._species:
            raise ValueError(f"Species {species} not found")
        return self._species.index(species)

    @property
    def species(self) -> list[Species]:
        """The species in the system, in the order they were added."""
        return list(self._species)

    @property
    def reactions(self) -> list[Reaction]:
        """The reactions in the system, in an arbitrary order."""
        return list(self._reactions)

    def get_reaction_matrices(self) -> ReactionMatrices:
        """Build the stochiometric matrices from the current reactions.

        Each reaction gives one row of ``reagents_m`` and ``products_m``, plus
        one entry of ``rates_v``. A bidirectional reaction contributes two
        rows: a forward one and a reverse one.
        """
        reagents_m = []
        products_m = []
        rates_v = []
        for r in self._reactions:
            r_s = r.reagents.stochiometry()
            p_s = r.products.stochiometry()
            r_row = np.zeros(len(self._species))
            p_row = np.zeros(len(self._species))
            for s, n in r_s:
                r_row[self.get_species_index(s)] = n
            for s, n in p_s:
                p_row[self.get_species_index(s)] = n
            reagents_m.append(r_row)
            products_m.append(p_row)
            rates_v.append(r.rate)
            if r.bidirectional:
                reagents_m.append(p_row)
                products_m.append(r_row)
                rates_v.append(r.rate)

        return ReactionMatrices(
            reagents_m=np.array(reagents_m),
            products_m=np.array(products_m),
            rates_v=np.array(rates_v),
        )

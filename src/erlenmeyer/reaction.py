from dataclasses import dataclass

import numpy as np

from erlenmeyer.symbols import AbstractReactionTerm, Species


@dataclass(frozen=True)
class Reaction:
    """A single chemical reaction with a rate constant.

    The reaction is written ``reagents => products``, or ``reagents <=>
    products`` when bidirectional. Passing ``None`` as the products makes it a
    decay: the reagents are consumed and nothing takes their place, written
    ``reagents => *``. This saves declaring an inert species just to hold what
    a decay leaves behind, which would otherwise grow without bound and take
    up a column of every result.

    Attributes
    ----------
    reagents : AbstractReactionTerm
        Term consumed by the reaction.
    products : AbstractReactionTerm or None
        Term produced by the reaction, or ``None`` for a decay.
    rate : float
        Rate constant of the reaction.
    bidirectional : bool, default False
        Whether the reverse reaction takes place as well, at the same rate.
        A decay can not be bidirectional, as it has nothing to react back.

    Raises
    ------
    ValueError
        If the reaction is bidirectional and has no products.
    """

    reagents: AbstractReactionTerm
    products: AbstractReactionTerm | None
    rate: float
    bidirectional: bool = False

    def __post_init__(self) -> None:
        """Reject a decay that is also marked bidirectional."""
        if (self.products is None) and self.bidirectional:
            raise ValueError("Can't have a bidirectional reaction with no products")

    def __repr__(self) -> str:
        """Write the reaction as ``reagents => products [rate]``.

        A bidirectional reaction uses ``<=>`` in place of ``=>``, and a decay
        writes its products as ``*``.
        """
        arrow = "<=>" if self.bidirectional else "=>"
        products = self.products if (self.products is not None) else "*"
        return f"{self.reagents} {arrow} {products} [{self.rate}]"


@dataclass(frozen=True)
class ReactionMatrices:
    """The stochiometric matrices of a set of reactions.

    Every reaction occupies one row of each matrix, and the columns follow the
    species order of the system the matrices were built from.

    Attributes
    ----------
    reagents_m : numpy.ndarray
        Amount of each species consumed by each reaction, shape
        ``[reactions, species]``.
    products_m : numpy.ndarray
        Amount of each species produced by each reaction, same shape. The row
        of a decay is all zeros.
    rates_v : numpy.ndarray
        Rate constant of each reaction, shape ``[reactions]``.
    """

    reagents_m: np.ndarray
    products_m: np.ndarray
    rates_v: np.ndarray


class ReactionSystem:
    """A collection of species and the reactions that take place among them.

    The species are given once, in the constructor, and their order fixes the
    column order of every array the system and its simulators return.
    Reactions are then added one at a time.
    """

    _species: list[Species]
    _reactions: set[Reaction]

    def __init__(self, species: list[Species]) -> None:
        """Build a system over the given species.

        Parameters
        ----------
        species : list of Species
            The species of the system, in the order that fixes the columns of
            its arrays.

        Raises
        ------
        ValueError
            If a species is given twice.
        """
        self._species = []
        # Check that none of these is present twice
        for s in species:
            if s in self._species:
                raise ValueError(f"Species {s} included twice")
            self._species.append(s)
        self._reactions = set()

    def add_reaction(self, reaction: Reaction) -> None:
        """Add a reaction to the system.

        Parameters
        ----------
        reaction : Reaction
            The reaction to add. Its species must all belong to the system,
            which is checked when the matrices are built.

        Raises
        ------
        ValueError
            If the reaction is already present.
        """
        if reaction in self._reactions:
            raise ValueError(f"Reaction {reaction} already present")
        self._reactions.add(reaction)

    def get_species_index(self, species: Species | str) -> int:
        """Return the index of a species in the system's species list.

        Parameters
        ----------
        species : Species or str
            The species, given either as a :class:`Species` or as its name.

        Returns
        -------
        int
            Index of the species, which is also its column in the system's
            arrays.

        Raises
        ------
        ValueError
            If the species does not belong to the system.
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
        rows: a forward one and a reverse one. A decay, which has no products,
        gives an all-zero row of ``products_m``, so its reagents simply leave
        the system.

        Returns
        -------
        ReactionMatrices
            The matrices of the reactions currently in the system. Their rows
            follow the arbitrary order of :attr:`reactions`.

        Raises
        ------
        ValueError
            If a reaction uses a species that does not belong to the system.
        """
        reagents_m = []
        products_m = []
        rates_v = []
        for r in self._reactions:
            r_s = r.reagents.stochiometry()
            if r.products is not None:
                p_s = r.products.stochiometry()
            else:
                p_s = []
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

import dataclasses

import numpy as np
import pytest

from erlenmeyer.reaction import Reaction, ReactionMatrices, ReactionSystem
from erlenmeyer.symbols import Species


class TestReactionConstruction:
    def test_reaction(self):
        h = Species("H")
        o = Species("O")
        h2o = Species("H2O")

        r = Reaction(2 * h + o, h2o, 1.0)

        assert r
        assert r.reagents.stochiometry() == [("H", 2), ("O", 1)]
        assert r.products.stochiometry() == [("H2O", 1)]
        assert r.rate == 1.0
        assert not r.bidirectional

        with pytest.raises(dataclasses.FrozenInstanceError):
            r.rate = 2  # type: ignore

        assert str(r) == "2H + O => H2O [1.0]"

    def test_has_expected_fields(self):
        h, o = Species("H"), Species("O")
        r = Reaction(h, o, 2.5)
        assert r.reagents == h
        assert r.products == o
        assert r.rate == 2.5


class TestReactionRepr:
    def test_default_arrow_is_forward(self):
        h, o = Species("H"), Species("O")
        assert str(Reaction(h, o, 1.0)) == "H => O [1.0]"

    def test_bidirectional_arrow(self):
        h, o = Species("H"), Species("O")
        assert str(Reaction(h, o, 1.0, bidirectional=True)) == "H <=> O [1.0]"

    def test_repr_matches_str(self):
        h, o = Species("H"), Species("O")
        assert repr(Reaction(h, o, 1.0)) == str(Reaction(h, o, 1.0))


class TestReactionEquality:
    def test_same_fields_are_equal(self):
        h, o = Species("H"), Species("O")
        assert Reaction(h, o, 1.0) == Reaction(h, o, 1.0)

    def test_different_rate(self):
        h, o = Species("H"), Species("O")
        assert Reaction(h, o, 1.0) != Reaction(h, o, 2.0)

    def test_different_bidirectional(self):
        h, o = Species("H"), Species("O")
        assert Reaction(h, o, 1.0) != Reaction(h, o, 1.0, bidirectional=True)


class TestReactionHashing:
    def test_equal_reactions_hash_equal(self):
        h, o = Species("H"), Species("O")
        assert hash(Reaction(h, o, 1.0)) == hash(Reaction(h, o, 1.0))

    def test_usable_in_sets(self):
        h, o = Species("H"), Species("O")
        reactions = {Reaction(h, o, 1.0), Reaction(h, o, 1.0), Reaction(h, o, 2.0)}
        assert len(reactions) == 2


class TestReactionSystemConstruction:
    def test_empty_system(self):
        system = ReactionSystem([])
        assert system.species == []
        assert system.reactions == []

    def test_species_are_stored_in_order(self):
        h, o, c = Species("H"), Species("O"), Species("C")
        system = ReactionSystem([h, o, c])
        assert system.species == [h, o, c]

    def test_duplicate_species_raise(self):
        h = Species("H")
        with pytest.raises(ValueError):
            ReactionSystem([h, h])

    def test_species_property_returns_a_copy(self):
        system = ReactionSystem([Species("H")])
        system.species.append(Species("O"))
        assert system.species == [Species("H")]


class TestReactionSystemAddReaction:
    def test_reaction_can_be_added(self):
        h, o = Species("H"), Species("O")
        system = ReactionSystem([h, o])
        r = Reaction(h, o, 1.0)
        system.add_reaction(r)
        assert system.reactions == [r]

    def test_duplicate_reaction_raises(self):
        h, o = Species("H"), Species("O")
        system = ReactionSystem([h, o])
        r = Reaction(h, o, 1.0)
        system.add_reaction(r)
        with pytest.raises(ValueError):
            system.add_reaction(r)


class TestReactionSystemGetSpeciesIndex:
    def test_index_of_known_species(self):
        h, o, c = Species("H"), Species("O"), Species("C")
        system = ReactionSystem([h, o, c])
        assert system.get_species_index(h) == 0
        assert system.get_species_index(o) == 1
        assert system.get_species_index(c) == 2

    def test_accepts_a_name_string(self):
        system = ReactionSystem([Species("H"), Species("O")])
        assert system.get_species_index("H") == 0
        assert system.get_species_index("O") == 1

    def test_unknown_species_raises(self):
        system = ReactionSystem([Species("H")])
        with pytest.raises(ValueError):
            system.get_species_index(Species("O"))
        with pytest.raises(ValueError):
            system.get_species_index("O")


class TestReactionSystemMatrices:
    def test_returns_a_reaction_matrices_instance(self):
        system = ReactionSystem([Species("H"), Species("O")])
        assert isinstance(
            system.get_reaction_matrices(), ReactionMatrices
        )

    def test_empty_system_gives_empty_matrices(self):
        system = ReactionSystem([])
        matrices = system.get_reaction_matrices()
        assert matrices.reagents_m.size == 0
        assert matrices.products_m.size == 0
        assert matrices.rates_v.size == 0

    def test_single_forward_reaction(self):
        h, o, h2o = Species("H"), Species("O"), Species("H2O")
        system = ReactionSystem([h, o, h2o])
        system.add_reaction(Reaction(2 * h + o, h2o, 3.0))
        matrices = system.get_reaction_matrices()
        assert matrices.reagents_m.tolist() == [[2.0, 1.0, 0.0]]
        assert matrices.products_m.tolist() == [[0.0, 0.0, 1.0]]
        assert matrices.rates_v.tolist() == [3.0]

    def test_bidirectional_reaction_gives_two_rows(self):
        h, o, h2o = Species("H"), Species("O"), Species("H2O")
        system = ReactionSystem([h, o, h2o])
        system.add_reaction(Reaction(2 * h + o, h2o, 2.0, bidirectional=True))
        matrices = system.get_reaction_matrices()
        # Forward row, then reverse row
        assert matrices.reagents_m.tolist() == [
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        assert matrices.products_m.tolist() == [
            [0.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
        assert matrices.rates_v.tolist() == [2.0, 2.0]

    def test_matrices_are_numpy_arrays(self):
        system = ReactionSystem([Species("H"), Species("O")])
        matrices = system.get_reaction_matrices()
        assert isinstance(matrices.reagents_m, np.ndarray)
        assert isinstance(matrices.products_m, np.ndarray)
        assert isinstance(matrices.rates_v, np.ndarray)

    def test_two_reactions_each_get_a_row(self):
        h, o, h2o = Species("H"), Species("O"), Species("H2O")
        system = ReactionSystem([h, o, h2o])
        forward = Reaction(2 * h + o, h2o, 1.0)
        reverse = Reaction(h2o, h + o, 0.5)
        system.add_reaction(forward)
        system.add_reaction(reverse)
        matrices = system.get_reaction_matrices()
        # Reactions come from a set, so row order is not deterministic.
        assert set(matrices.rates_v.tolist()) == {0.5, 1.0}
        forward_row_reagents = [2.0, 1.0, 0.0]
        forward_row_products = [0.0, 0.0, 1.0]
        reverse_row_reagents = [0.0, 0.0, 1.0]
        reverse_row_products = [1.0, 1.0, 0.0]
        reagents = matrices.reagents_m.tolist()
        products = matrices.products_m.tolist()
        forward_index = reagents.index(forward_row_reagents)
        assert products[forward_index] == forward_row_products
        reverse_index = reagents.index(reverse_row_reagents)
        assert products[reverse_index] == reverse_row_products
        assert {tuple(row) for row in reagents} == {
            tuple(forward_row_reagents),
            tuple(reverse_row_reagents),
        }

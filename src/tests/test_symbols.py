import pytest

from erlenmeyer.symbols import (
    AbstractReactionTerm,
    ReactionTerm,
    SingleReactionTerm,
    Species,
)


class TestSingleReactionTermConstruction:
    """Validation and storage of the species/amount pair."""

    def test_default_amount_is_one(self):
        c = SingleReactionTerm("C")
        assert c.species == "C"
        assert c.amount == 1

    def test_explicit_amount(self):
        c = SingleReactionTerm("C", 2)
        assert c.species == "C"
        assert c.amount == 2

    @pytest.mark.parametrize("name", ["H", "h", "He", "Na", "X1", "Ab12Cd34", "x9"])
    def test_valid_species_names(self, name):
        assert SingleReactionTerm(name).species == name

    @pytest.mark.parametrize("name", ["H_2", "H_", "Na_Cl", "a_b_c", "X1_y2_"])
    def test_underscores_are_allowed_after_the_first_character(self, name):
        assert SingleReactionTerm(name).species == name

    @pytest.mark.parametrize(
        "name", ["", "1H", "9", "_H", "_", "H-", "H+", "Na Cl", "H.", "H2O!", "H\n"]
    )
    def test_invalid_species_names(self, name):
        with pytest.raises(AssertionError):
            SingleReactionTerm(name)

    @pytest.mark.parametrize("name", [None, 1, 2.0, {}, [], ("H", 1)])
    def test_species_must_be_a_string(self, name):
        with pytest.raises(AssertionError):
            SingleReactionTerm(name)  # type: ignore[arg-type]

    @pytest.mark.parametrize("amount", [0, -1, -10])
    def test_amount_must_be_positive(self, amount):
        with pytest.raises(AssertionError):
            SingleReactionTerm("C", amount)

    @pytest.mark.parametrize("amount", [3.2, "2", None, [2], True, False])
    def test_amount_must_be_an_integer(self, amount):
        # Booleans are integers in Python, but they are not valid amounts
        with pytest.raises(AssertionError):
            SingleReactionTerm("C", amount)  # type: ignore[arg-type]

    def test_properties_are_read_only(self):
        c = SingleReactionTerm("C", 2)
        with pytest.raises(AttributeError):
            c.species = "N"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            c.amount = 3  # type: ignore[misc]


class TestSingleReactionTermStochiometry:
    def test_stochiometry_is_a_single_pair(self):
        assert SingleReactionTerm("C", 2).stochiometry() == [("C", 2)]

    def test_stochiometry_of_default_amount(self):
        assert SingleReactionTerm("C").stochiometry() == [("C", 1)]

    def test_is_an_abstract_reaction_term(self):
        assert isinstance(SingleReactionTerm("C"), AbstractReactionTerm)


class TestSingleReactionTermRepr:
    def test_amount_one_is_omitted(self):
        assert repr(SingleReactionTerm("H", 1)) == "H"

    def test_amount_above_one_is_prefixed(self):
        assert repr(SingleReactionTerm("H", 2)) == "2H"

    def test_str_matches_repr(self):
        h3 = SingleReactionTerm("H", 3)
        assert str(h3) == repr(h3) == "3H"


class TestSingleReactionTermEquality:
    def test_equal_species_and_amount(self):
        assert SingleReactionTerm("H", 2) == SingleReactionTerm("H", 2)

    def test_different_amount(self):
        assert SingleReactionTerm("H", 2) != SingleReactionTerm("H", 3)

    def test_different_species(self):
        assert SingleReactionTerm("H", 2) != SingleReactionTerm("O", 2)

    def test_species_names_are_case_sensitive(self):
        assert SingleReactionTerm("h") != SingleReactionTerm("H")

    def test_species_equals_equivalent_single_term(self):
        assert Species("H") == SingleReactionTerm("H", 1)
        assert SingleReactionTerm("H", 1) == Species("H")

    def test_equals_a_compound_term_with_the_same_stochiometry(self):
        # Terms are compared by stochiometry, not by class
        assert SingleReactionTerm("H", 2) == ReactionTerm([("H", 2)])
        assert SingleReactionTerm("H", 2) != ReactionTerm([("H", 2), ("O", 1)])

    @pytest.mark.parametrize("other", [1, "H", None, ("H", 1), [("H", 1)]])
    def test_comparison_with_unrelated_types_raises(self, other):
        with pytest.raises(TypeError):
            assert SingleReactionTerm("H") == other  # type: ignore[comparison-overlap]


class TestSingleReactionTermHashing:
    def test_equal_terms_hash_equal(self):
        assert hash(SingleReactionTerm("H", 2)) == hash(SingleReactionTerm("H", 2))

    def test_hash_follows_the_stochiometry(self):
        assert hash(SingleReactionTerm("H", 2)) == hash((("H", 2),))

    def test_usable_in_sets_and_dicts(self):
        terms = {SingleReactionTerm("H", 2), SingleReactionTerm("H", 2), Species("O")}
        assert len(terms) == 2
        counts = {Species("H"): 1, Species("O"): 2}
        assert counts[SingleReactionTerm("H", 1)] == 1  # type: ignore[index]


class TestSingleReactionTermAddition:
    def test_same_species_amounts_add(self):
        h3 = SingleReactionTerm("H", 1) + SingleReactionTerm("H", 2)
        assert isinstance(h3, SingleReactionTerm)
        assert h3 == SingleReactionTerm("H", 3)

    def test_addition_does_not_mutate_the_operands(self):
        h = SingleReactionTerm("H", 1)
        h + SingleReactionTerm("H", 2)  # type: ignore
        assert h.amount == 1

    def test_different_species_give_a_compound_term(self):
        term = SingleReactionTerm("H", 2) + SingleReactionTerm("O", 1)
        assert isinstance(term, ReactionTerm)
        assert term.stochiometry() == [("H", 2), ("O", 1)]

    def test_addition_of_different_species_is_commutative(self):
        left = SingleReactionTerm("O", 1) + SingleReactionTerm("H", 2)
        right = SingleReactionTerm("H", 2) + SingleReactionTerm("O", 1)
        assert left == right

    def test_addition_to_a_compound_term(self):
        h2o = SingleReactionTerm("H", 2) + SingleReactionTerm("O", 1)
        term = SingleReactionTerm("C", 1) + h2o
        assert isinstance(term, ReactionTerm)
        assert term.stochiometry() == [("C", 1), ("H", 2), ("O", 1)]

    def test_addition_to_a_compound_term_merges_like_species(self):
        h2o = SingleReactionTerm("H", 2) + SingleReactionTerm("O", 1)
        assert (Species("H") + h2o).stochiometry() == [("H", 3), ("O", 1)]

    @pytest.mark.parametrize("other", [1, "H", None, ("H", 1), [("H", 1)]])
    def test_addition_with_unsupported_types_raises(self, other):
        with pytest.raises(TypeError):
            SingleReactionTerm("H") + other  # type: ignore[operator]

    def test_no_reflected_addition(self):
        with pytest.raises(TypeError):
            1 + SingleReactionTerm("H")  # type: ignore[operator]


class TestSpecies:
    def test_amount_is_always_one(self):
        h = Species("H")
        assert h.species == "H"
        assert h.amount == 1
        assert str(h) == "H"

    def test_is_a_single_reaction_term(self):
        assert isinstance(Species("H"), SingleReactionTerm)

    def test_same_validation_as_single_reaction_term(self):
        with pytest.raises(AssertionError):
            Species("2H")
        with pytest.raises(AssertionError):
            Species(3)  # type: ignore[arg-type]

    def test_takes_no_amount_argument(self):
        with pytest.raises(TypeError):
            Species("H", 2)  # type: ignore[call-arg]

    def test_multiplication_on_the_right(self):
        h2 = Species("H") * 2
        assert isinstance(h2, SingleReactionTerm)
        assert h2 == SingleReactionTerm("H", 2)

    def test_multiplication_on_the_left(self):
        assert 2 * Species("H") == Species("H") * 2

    def test_multiplication_gives_a_plain_single_term(self):
        h2 = 2 * Species("H")
        assert not isinstance(h2, Species)

    def test_multiplication_by_one(self):
        assert 1 * Species("H") == Species("H")

    @pytest.mark.parametrize("factor", [0, -1])
    def test_multiplication_by_a_non_positive_number_raises(self, factor):
        with pytest.raises(AssertionError):
            factor * Species("H")  # type: ignore[operator]

    @pytest.mark.parametrize("factor", [2.0, "2", None, True, Species("O")])
    def test_multiplication_by_a_non_integer_raises(self, factor):
        with pytest.raises(TypeError):
            Species("H") * factor  # type: ignore[operator]

    def test_multiplication_does_not_mutate_the_species(self):
        h = Species("H")
        _ = 5 * h
        assert h.amount == 1


class TestReactionTermConstruction:
    def test_from_single_reaction_terms(self):
        term = ReactionTerm([SingleReactionTerm("H", 2), Species("O")])
        assert term.stochiometry() == [("H", 2), ("O", 1)]

    def test_from_tuples(self):
        term = ReactionTerm([("H", 2), ("O", 1)])
        assert term.stochiometry() == [("H", 2), ("O", 1)]

    def test_from_mixed_inputs(self):
        term = ReactionTerm([("H", 2), Species("O")])
        assert term.stochiometry() == [("H", 2), ("O", 1)]

    def test_from_single_element_tuples(self):
        # The amount defaults to 1, as in SingleReactionTerm
        term = ReactionTerm([("H",), ("O",)])  # type: ignore[list-item]
        assert term.stochiometry() == [("H", 1), ("O", 1)]

    def test_terms_are_sorted(self):
        term = ReactionTerm([("O", 1), ("C", 3), ("H", 2)])
        assert term.stochiometry() == [("C", 3), ("H", 2), ("O", 1)]

    def test_repeated_species_are_merged(self):
        term = ReactionTerm([("H", 3), ("O", 1), ("H", 1)])
        assert term.stochiometry() == [("H", 4), ("O", 1)]

    def test_empty_term(self):
        assert ReactionTerm([]).stochiometry() == []

    def test_tuples_are_validated(self):
        with pytest.raises(AssertionError):
            ReactionTerm([("H", 0)])
        with pytest.raises(AssertionError):
            ReactionTerm([("2H", 1)])

    def test_malformed_tuples_raise(self):
        with pytest.raises(TypeError):
            ReactionTerm([("H", 1, 2)])  # type: ignore[list-item]
        with pytest.raises(TypeError):
            ReactionTerm([ReactionTerm([("H", 1)])])  # type: ignore[list-item]

    def test_does_not_keep_a_reference_to_the_input_list(self):
        terms = [SingleReactionTerm("H", 2)]
        rterm = ReactionTerm(terms)
        terms.append(SingleReactionTerm("O", 1))
        assert rterm.stochiometry() == [("H", 2)]

    def test_is_an_abstract_reaction_term(self):
        assert isinstance(ReactionTerm([]), AbstractReactionTerm)


class TestReactionTermStochiometry:
    def test_stochiometry_lists_every_term(self):
        term = ReactionTerm([("C", 1), ("H", 4)])
        assert term.stochiometry() == [("C", 1), ("H", 4)]

    def test_species_can_be_read_back(self):
        term = ReactionTerm([("C", 1), ("H", 4), ("O", 2)])
        assert [s for s, _ in term.stochiometry()] == ["C", "H", "O"]
        assert [n for _, n in term.stochiometry()] == [1, 4, 2]


class TestReactionTermRepr:
    def test_terms_are_joined_by_plus_signs(self):
        assert repr(ReactionTerm([("H", 2), ("O", 1)])) == "2H + O"

    def test_empty_term(self):
        assert repr(ReactionTerm([])) == "0"

    def test_str_matches_repr(self):
        term = ReactionTerm([("C", 1), ("H", 4)])
        assert str(term) == repr(term) == "C + 4H"


class TestReactionTermEquality:
    def test_same_stochiometry(self):
        assert ReactionTerm([("H", 2), ("O", 1)]) == ReactionTerm([("H", 2), ("O", 1)])

    def test_order_of_construction_does_not_matter(self):
        assert ReactionTerm([("O", 1), ("H", 2)]) == ReactionTerm([("H", 2), ("O", 1)])

    def test_different_stochiometry(self):
        assert ReactionTerm([("H", 2), ("O", 1)]) != ReactionTerm([("H", 2), ("O", 2)])
        assert ReactionTerm([("H", 2)]) != ReactionTerm([("H", 2), ("O", 1)])

    def test_empty_terms_are_equal(self):
        assert ReactionTerm([]) == ReactionTerm([])

    @pytest.mark.parametrize("other", [1, "H", None, ("H", 1), [("H", 1)]])
    def test_comparison_with_unrelated_types_raises(self, other):
        with pytest.raises(TypeError):
            assert ReactionTerm([("H", 2)]) == other  # type: ignore[comparison-overlap]


class TestReactionTermHashing:
    def test_equal_terms_hash_equal(self):
        left = ReactionTerm([("O", 1), ("H", 2)])
        right = ReactionTerm([("H", 2), ("O", 1)])
        assert hash(left) == hash(right)

    def test_matches_the_hash_of_an_equivalent_single_term(self):
        assert hash(ReactionTerm([("H", 2)])) == hash(SingleReactionTerm("H", 2))

    def test_usable_in_sets_and_dicts(self):
        terms = {ReactionTerm([("H", 2), ("O", 1)]), ReactionTerm([("O", 1), ("H", 2)])}
        assert len(terms) == 1
        assert {ReactionTerm([("H", 2), ("O", 1)]): "water"}[
            ReactionTerm([("O", 1), ("H", 2)])
        ] == "water"


class TestReactionTermAddition:
    def test_addition_of_two_compound_terms(self):
        term = ReactionTerm([("H", 2), ("O", 1)]) + ReactionTerm([("C", 1), ("H", 1)])
        assert term.stochiometry() == [("C", 1), ("H", 3), ("O", 1)]

    def test_addition_of_a_single_term(self):
        term = ReactionTerm([("H", 2), ("O", 1)]) + Species("C")
        assert term.stochiometry() == [("C", 1), ("H", 2), ("O", 1)]

    def test_addition_merges_like_species(self):
        term = ReactionTerm([("H", 2), ("O", 1)]) + Species("O")
        assert term.stochiometry() == [("H", 2), ("O", 2)]

    def test_addition_is_commutative(self):
        h2o = ReactionTerm([("H", 2), ("O", 1)])
        assert h2o + Species("C") == Species("C") + h2o

    def test_addition_does_not_mutate_the_operands(self):
        h2o = ReactionTerm([("H", 2), ("O", 1)])
        h2o + Species("O")  # type: ignore
        assert h2o.stochiometry() == [("H", 2), ("O", 1)]

    def test_result_collapses_to_a_single_term(self):
        term = ReactionTerm([("H", 2)]) + Species("H")
        assert isinstance(term, SingleReactionTerm)
        assert term == SingleReactionTerm("H", 3)

    def test_addition_of_an_empty_term(self):
        h2o = ReactionTerm([("H", 2), ("O", 1)])
        assert h2o + ReactionTerm([]) == h2o

    @pytest.mark.parametrize("other", [1, "H", None, ("H", 1), [("H", 1)]])
    def test_addition_with_unsupported_types_raises(self, other):
        with pytest.raises(TypeError):
            ReactionTerm([("H", 2)]) + other  # type: ignore[operator]


class TestAbstractReactionTerm:
    def test_can_not_be_instantiated(self):
        with pytest.raises(TypeError):
            AbstractReactionTerm()  # type: ignore[abstract]

    def test_subclass_must_implement_stochiometry(self):
        class Incomplete(AbstractReactionTerm):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_complete_subclass_inherits_the_operators(self):
        class Complete(AbstractReactionTerm):
            def stochiometry(self):
                return [("H", 1)]

        assert Complete().stochiometry() == [("H", 1)]
        assert repr(Complete()) == "H"
        assert Complete() == Species("H")
        assert Complete() + Species("H") == SingleReactionTerm("H", 2)


class TestBuildingFormulas:
    """End to end use of the operators, as a caller would write them."""

    def test_water(self):
        h, o = Species("H"), Species("O")
        h2o = 2 * h + o
        assert isinstance(h2o, ReactionTerm)
        assert h2o.stochiometry() == [("H", 2), ("O", 1)]
        assert repr(h2o) == "2H + O"

    def test_repeated_addition_of_the_same_species(self):
        h = Species("H")
        assert h + h == 2 * h
        assert h + h + h == 3 * h

    def test_three_species_chained_from_the_left(self):
        c, h, o = Species("C"), Species("H"), Species("O")
        glucose = 6 * c + 12 * h + 6 * o
        assert glucose.stochiometry() == [("C", 6), ("H", 12), ("O", 6)]

    def test_grouping_does_not_change_the_result(self):
        c, h, o = Species("C"), Species("H"), Species("O")
        assert (6 * c + 12 * h) + 6 * o == 6 * c + (12 * h + 6 * o)

    def test_like_species_merge_wherever_they_appear(self):
        h, o, na = Species("H"), Species("O"), Species("Na")
        assert (h + o + h + na).stochiometry() == [("H", 2), ("Na", 1), ("O", 1)]

    def test_underscored_names_can_be_combined(self):
        assert repr(2 * Species("H_aq") + Species("O_g")) == "2H_aq + O_g"

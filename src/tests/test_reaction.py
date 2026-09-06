import dataclasses

import pytest

from erlenmeyer.reaction import Reaction
from erlenmeyer.symbols import Species


def test_reaction():
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
        r.rate = 2 # type: ignore

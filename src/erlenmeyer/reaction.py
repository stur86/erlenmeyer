from dataclasses import dataclass
from erlenmeyer.symbols import AbstractReactionTerm, Stochiometry

@dataclass(frozen=True)
class Reaction:
    reagents: AbstractReactionTerm
    products: AbstractReactionTerm
    rate: float
    bidirectional: bool = False

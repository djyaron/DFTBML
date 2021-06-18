from __future__ import annotations

import re
from typing import ItemsView, KeysView, List, ValuesView

from consts import ALIAS2TARGET


class Target:
    def __init__(self, target: str) -> None:
        r"""Convert target string in natural language into a dictionary"""
        self.tt = target
        self.target = self.interpret(self.tt)

    def __len__(self) -> int:
        return self.target.__len__()

    def items(self) -> ItemsView:
        return self.target.items()

    def keys(self) -> KeysView:
        return self.target.keys()

    def operators(self) -> List[int]:
        return list(self.target.values())

    def targets(self) -> List[str]:
        return list(self.target.keys())

    def values(self) -> ValuesView:
        return self.target.values()

    @staticmethod
    def interpret(target: str) -> dict:
        tg = re.findall(r'[a-z]+', target)  # find targets aliases (alphabetical)
        op = re.findall(r'[+-]', target)  # find operators (addition/subtraction)
        # Sanity check
        try:
            assert len(tg) == len(op) or len(tg) - len(op) == 1
            for t in tg:
                assert t in ALIAS2TARGET.keys()
        except AssertionError:
            raise ValueError("Invalid target")
        # Ensure same number of targets/operators
        if len(tg) == len(op):  # e.g. '+fm-pf+pr' or '+ fm - pf + pr'
            pass
        else:  # e.g. 'fm-pf+pr' or 'fm - pf + pr'
            op.insert(0, '+')
        # Convert to dictionary
        tg = [ALIAS2TARGET[t] for t in tg]
        op = [1 if o == '+' else -1 for o in op]
        return dict(zip(tg, op))


class Constraint(Target):
    @staticmethod
    def interpret(constr: str) -> dict:
        ct = re.findall(r'[0-9]+', constr)
        op = re.findall(r'[+-]', constr)
        # Sanity check
        try:
            assert len(ct) == len(op) or len(ct) - len(op) == 1
        except AssertionError:
            raise ValueError("Invalid target")
        # Ensure same number of constraints/operators
        if len(ct) == len(op):  # e.g. '+2-1' or '+ 2 - 1'
            pass
        else:  # e.g. '2-1' or '2 - 1'
            op.insert(0, '+')
        # Convert to dictionary
        ct = [int(c) for c in ct]
        op = [1 if o == '+' else -1 for o in op]
        return dict(zip(ct, op))


if __name__ == '__main__':
    t1 = Target('fm-pf+pr')
    c1 = Constraint('-1+2')

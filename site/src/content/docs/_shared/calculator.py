"""Shared calculator tool used by documentation examples.

Included into pages via the mkdocs-snippets syntax, e.g.

    --8<-- "_shared/calculator.py:calculator"

Kept deliberately small and dependency-free so every example stays
copy-pasteable, and restricted to arithmetic so a docs example never
demonstrates evaluating model-supplied input as code.
"""

# --8<-- [start:calculator]
import ast
import operator

from strands import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as "144 ** 0.5" or "450 / 120".

    Args:
        expression: The arithmetic expression to evaluate.
    """

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base, exponent = ev(node.left), ev(node.right)
            if abs(exponent) > 64:  # an unbounded pow hangs uninterruptibly
                raise ValueError(f"exponent too large: {exponent}")
            return base**exponent
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"not arithmetic (+ - * / ** and parens only): {expression!r}")

    return str(ev(ast.parse(expression, mode="eval").body))
# --8<-- [end:calculator]

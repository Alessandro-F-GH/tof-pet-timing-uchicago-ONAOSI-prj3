"""Plain-text escaping shared by report table exporters."""
from typing import Any


def escape_latex(value: Any) -> str:
    """Escape plain text, including file names, without interpreting TeX commands."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))

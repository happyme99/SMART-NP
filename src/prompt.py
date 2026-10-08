"""Prompt construction for descriptor-selected molecular retrieval."""

from __future__ import annotations

from descriptor import format_descriptors
from retrieval import RetrievedExample


def build_prompt(smiles: str, examples: list[RetrievedExample], selected_descriptors: list[str]) -> tuple[str, str]:
    system = "\n".join([
        "You are a world-class natural product chemist and structural analyst.",
        "Use the query, selected descriptors, and provided train examples.",
        "Choose exactly one class name. Do not invent or generalize class names.",
    ])
    lines = ["Query natural product:", f"- SMILES: {smiles}", format_descriptors(smiles, selected_descriptors), ""]
    for example in examples:
        lines.extend([
            f"[Retrieved Example {example.rank}]",
            f"- class name: {example.class_name}",
            f"- SMILES: {example.smiles}",
            format_descriptors(example.smiles, selected_descriptors),
            "",
        ])
    lines.extend(["Guidance:", "- Compare query with all retrieved examples using available evidence.", "- Choose class best supported by evidence.", "Return one line only: exactly one class name."])
    return system, "\n".join(lines)

"""Structural descriptors used by prompts."""

from __future__ import annotations

from collections import deque

from rdkit import Chem
from rdkit.Chem import Fragments, rdMolDescriptors


BASE_LABELS = {
    "carbon_count": "number of carbon",
    "nitrogen_count": "number of nitrogen (N)",
    "oxygen_count": "number of oxygen (O)",
    "sulfur_count": "number of sulfur (S)",
    "phosphorus_count": "number of phosphorus (P)",
    "halogen_count": "number of halogens",
    "stereocenter_count": "number of stereocenters",
    "ring_count": "number of rings",
    "aromatic_ring_count": "number of aromatic rings",
    "aliphatic_ring_count": "number of aliphatic rings",
    "saturated_ring_count": "number of saturated rings",
    "fused_ring_count": "fused ring count",
    "largest_fused_ring_system_size": "largest fused ring system size",
    "isolated_ring_system_count": "isolated ring system count",
    "spiro_atom_count": "number of spiro atoms",
    "bridgehead_atom_count": "number of bridgehead atoms",
    "ring_sizes": "ring sizes",
    "nitrogen_in_ring_count": "number of nitrogen in rings",
    "nitrogen_side_chain_count": "number of nitrogen in side chains",
    "longest_acyclic_carbon_chain": "longest acyclic carbon chain",
    "exocyclic_double_bond_count": "exocyclic double bond count",
    "quaternary_carbon_count": "quaternary carbon count",
    "sugar_ring_count": "sugar ring count",
    "glycosidic_bond_count": "glycosidic bond count",
}
FRAGMENTS = {name: getattr(Fragments, name) for name in sorted(dir(Fragments)) if name.startswith("fr_") and callable(getattr(Fragments, name))}
DESCRIPTOR_KEYS = tuple(BASE_LABELS) + tuple(FRAGMENTS)
DESCRIPTOR_LABELS = {**BASE_LABELS, **{key: key for key in FRAGMENTS}}


def _fused_components(mol):
    rings = [set(map(int, ring)) for ring in Chem.GetSymmSSSR(mol)]
    graph = {index: set() for index in range(len(rings))}
    for left, ring in enumerate(rings):
        for right in range(left + 1, len(rings)):
            if len(ring & rings[right]) >= 2:
                graph[left].add(right)
                graph[right].add(left)
    seen, components = set(), []
    for start in graph:
        if start in seen:
            continue
        stack, component = [start], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(graph[node] - seen)
        components.append(component)
    return components


def _sugar_rings(mol):
    out = []
    for ring in Chem.GetSymmSSSR(mol):
        indices = set(map(int, ring))
        atoms = [mol.GetAtomWithIdx(index) for index in indices]
        if len(indices) not in {5, 6} or any(atom.GetIsAromatic() for atom in atoms):
            continue
        carbons = [atom for atom in atoms if atom.GetAtomicNum() == 6]
        if sum(atom.GetAtomicNum() == 8 for atom in atoms) != 1 or len(carbons) != len(indices) - 1:
            continue
        oxygens = {neighbor.GetIdx() for atom in carbons for neighbor in atom.GetNeighbors() if neighbor.GetIdx() not in indices and neighbor.GetAtomicNum() == 8}
        if len(oxygens) >= 2:
            out.append(indices)
    return out


def _longest_acyclic_carbon_chain(mol):
    nodes = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
    graph = {node: [] for node in nodes}
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if left in graph and right in graph and not bond.IsInRing():
            graph[left].append(right)
            graph[right].append(left)
    def farthest(start):
        queue, seen, best = deque([(start, 1)]), {start}, (start, 1)
        while queue:
            node, distance = queue.popleft()
            if distance > best[1]:
                best = node, distance
            for next_node in graph[node]:
                if next_node not in seen:
                    seen.add(next_node)
                    queue.append((next_node, distance + 1))
        return best, seen
    seen, longest = set(), 0
    for node in nodes:
        if node in seen:
            continue
        (edge, distance), component = farthest(node)
        seen.update(component)
        longest = max(longest, farthest(edge)[0][1], distance)
    return longest


def _glycosidic_bonds(mol, sugar_rings):
    bonds = set()
    for ring in sugar_rings:
        ring_oxygen = {index for index in ring if mol.GetAtomWithIdx(index).GetAtomicNum() == 8}
        for index in ring:
            atom = mol.GetAtomWithIdx(index)
            if atom.GetAtomicNum() != 6 or not any(neighbor.GetIdx() in ring_oxygen for neighbor in atom.GetNeighbors()):
                continue
            for bond in atom.GetBonds():
                neighbor = bond.GetOtherAtom(atom)
                if bond.GetBondType() == Chem.BondType.SINGLE and neighbor.GetIdx() not in ring and neighbor.GetAtomicNum() == 8:
                    if any(other.GetIdx() != atom.GetIdx() and other.GetAtomicNum() > 1 for other in neighbor.GetNeighbors()):
                        bonds.add(tuple(sorted((atom.GetIdx(), neighbor.GetIdx()))))
    return len(bonds)


def compute_descriptors(smiles: str) -> dict[str, int | list[int]]:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    rings = [set(map(int, ring)) for ring in Chem.GetSymmSSSR(mol)]
    components = _fused_components(mol)
    fused = [len(component) for component in components if len(component) >= 2]
    sugars = _sugar_rings(mol)
    nitrogen_in_ring = sum(atom.GetAtomicNum() == 7 and atom.IsInRing() for atom in mol.GetAtoms())
    descriptors = {
        "carbon_count": sum(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()),
        "nitrogen_count": sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()),
        "oxygen_count": sum(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()),
        "sulfur_count": sum(atom.GetAtomicNum() == 16 for atom in mol.GetAtoms()),
        "phosphorus_count": sum(atom.GetAtomicNum() == 15 for atom in mol.GetAtoms()),
        "halogen_count": sum(atom.GetAtomicNum() in {9, 17, 35, 53} for atom in mol.GetAtoms()),
        "stereocenter_count": int(rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "aliphatic_ring_count": int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "saturated_ring_count": int(rdMolDescriptors.CalcNumSaturatedRings(mol)),
        "fused_ring_count": int(sum(fused)),
        "largest_fused_ring_system_size": int(max(fused, default=0)),
        "isolated_ring_system_count": int(sum(len(component) == 1 for component in components)),
        "spiro_atom_count": int(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
        "bridgehead_atom_count": int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        "ring_sizes": sorted(map(len, rings)),
        "nitrogen_in_ring_count": int(nitrogen_in_ring),
        "nitrogen_side_chain_count": int(sum(atom.GetAtomicNum() == 7 and not atom.IsInRing() for atom in mol.GetAtoms())),
        "longest_acyclic_carbon_chain": int(_longest_acyclic_carbon_chain(mol)),
        "exocyclic_double_bond_count": sum(bond.GetBondType() == Chem.BondType.DOUBLE and not bond.IsInRing() and bond.GetBeginAtom().IsInRing() != bond.GetEndAtom().IsInRing() for bond in mol.GetBonds()),
        "quaternary_carbon_count": sum(atom.GetAtomicNum() == 6 and atom.GetDegree() == 4 and sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) == 4 for atom in mol.GetAtoms()),
        "sugar_ring_count": len(sugars),
        "glycosidic_bond_count": _glycosidic_bonds(mol, sugars),
    }
    descriptors.update({name: int(function(mol, countUnique=True)) for name, function in FRAGMENTS.items()})
    return descriptors


def format_descriptors(smiles: str, keys: list[str]) -> str:
    values = compute_descriptors(smiles)
    lines = ["Structural descriptors used for comparison:"]
    if not keys:
        return "\n".join(lines + ["- (none)"])
    for key in keys:
        value = values[key]
        lines.append(f"- {DESCRIPTOR_LABELS.get(key, key)}: {value}")
    return "\n".join(lines)

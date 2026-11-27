#!/usr/bin/env python3
"""The packing scenario of Appendix A, evaluated for lipoxygenase's own donor and acceptor.

scripts/packing_bound.py states the scenario for a generic methine donor: three heavy-atom
substituents at d_bond = 2.15 A and an acceptor heavy atom at 2.0 A. Linoleate C11 is a methylene
with two heavy-atom substituents, and the acceptor oxygen of the clamped geometries lies near 2.42 A.
This script evaluates the same construction with those two inputs, using the published functions
unchanged, and compares both versions with the all-protein share of the fourteen clamped geometries
(results/confinement_budget.json: other protein + iron first shell + cavity backbone + cavity side
chain). Writes results/audit_2026_11/packing_slo.json.
Usage: audit_packing_slo.py
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('packing_bound', ROOT / 'scripts/packing_bound.py')
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)

GRID = (2.9, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5)      # the published d_pack grid
PROTEIN = ('other protein', 'cofactor first shell', 'JBC cavity backbone', 'JBC cavity side chain')


def scenario(n_bond, d_acc):
    rows = [(dp, pb.n_max(dp), pb.cavity_fraction(dp, n_pack=pb.n_max(dp), n_bond=n_bond, d_acc=d_acc))
            for dp in GRID]
    typical = pb.cavity_fraction(3.4, n_pack=pb.n_max(3.4), n_bond=n_bond, d_acc=d_acc)
    worst = max(rows, key=lambda r: r[2])
    return dict(n_bond=n_bond, d_acc_A=d_acc, typical_percent=100 * typical,
                worst_percent=100 * worst[2], worst_d_pack_A=worst[0], worst_n_pack=worst[1])


def main():
    budget = json.loads((ROOT / 'results/confinement_budget.json').read_text())
    share = {k: 100 * sum(v['categories'][c]['fraction'] for c in PROTEIN) for k, v in budget.items()}
    out = dict(d_bond_A=pb.d_bond(), all_protein_percent=share,
               generic=scenario(3, 2.0), slo_methylene=scenario(2, 2.42))
    for key in ('generic', 'slo_methylene'):
        s = out[key]
        s['n_exceed_typical'] = sum(x > s['typical_percent'] for x in share.values())
        s['n_exceed_worst'] = sum(x > s['worst_percent'] for x in share.values())
    out['n_geometries'] = len(share)
    out['max_all_protein_percent'] = max(share.values())
    dest = ROOT / 'results/audit_2026_11/packing_slo.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    for key in ('generic', 'slo_methylene'):
        s = out[key]
        print(f"{key:14s} typical {s['typical_percent']:.2f} %  worst {s['worst_percent']:.2f} % "
              f"(d_pack {s['worst_d_pack_A']}, {s['worst_n_pack']} atoms)  exceed: {s['n_exceed_typical']} and "
              f"{s['n_exceed_worst']} of {len(share)}")
    print(f"largest all-protein share {out['max_all_protein_percent']:.2f} %; wrote {dest}")


if __name__ == '__main__':
    main()

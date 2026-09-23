"""Chemically valid QM cluster builder for the SLO PCET active site.

REPLACES the cluster_atoms() selection in proton_pes_scanner.py, which was
found on 2026-08-10 to emit a cluster with 35 severed bonds and ZERO hydrogen
caps (the substrate entered as an 18-carbon skeleton carrying 2 hydrogens,
missing 29). That construction is not a molecule; it is a set of dangling
valences, and it produced the spin contamination (<S^2> ~ 11-12.3 against 6.0
requested) seen in BOTH ORCA and NWChem, the inflated ~73 kcal/mol apparent
barriers, and the CDFT convergence pathology.

DESIGN PRINCIPLE: every bond cut at a fragment boundary is closed with a
hydrogen cap placed along the original bond vector at a standard X-H length.
The builder then VALIDATES the result and raises if any heavy atom is left
under-coordinated. An invalid cluster is never returned.

FRAGMENTS (SLO-1 numbering as it appears in this topology):

  Fe1818          Fe(III), high-spin d5, S = 5/2                    charge +3
  OH1819          hydroxide acceptor                                charge -1
  HD1477 }
  HD2482 }        three His imidazole ligands, cut CB-CA,
  HD3668 }        capped -> 4-methylimidazole, neutral              charge  0
  ASN672          side-chain amide, cut CB-CA,
                  capped -> acetamide, neutral                      charge  0
  IE1817          C-terminal Ile: side chain + carboxylate via CA,
                  cut CA-N, capped -> 3-methylpentanoate            charge -1
  LIG820          substrate 1,4-pentadiene unit
                  C10=C13-C14(H2)-C17=C15, cut C10-C6 and C15-C11,
                  capped -> penta-1,4-diene, neutral                charge  0
                                                                    ---------
                                                          NET CHARGE      +1

  sum(Z) = 300  ->  charge +1 gives 299 electrons (ODD)
                ->  mult 6 (5 unpaired, S = 5/2) is allowed, <S^2> = 8.75

The substrate carboxylate head group is deliberately EXCLUDED: it is ~15 A
from the reactive centre, carries its own -1, and its inclusion in the old
builder (as two bare oxygens with no hydrogens anywhere on the chain) was one
source of the broken valence count.
"""
import numpy as np
from collections import defaultdict

# Standard X-H bond lengths for link-atom capping (Angstrom)
CAP_LENGTH = {'C': 1.09, 'N': 1.01, 'O': 0.96, 'S': 1.34}

# Expected heavy-atom valences (number of sigma-bonded neighbours)
EXPECTED_VALENCE = {'C': 4, 'N': 3, 'O': 2, 'S': 2}

ATOMIC_NUMBER = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'Fe': 26, 'P': 15}

# --- fragment definitions -------------------------------------------------
HIS_KEEP = ['CB', 'HB2', 'HB3', 'CG', 'ND1', 'HD1',
            'CE1', 'HE1', 'NE2', 'HE2', 'CD2', 'HD2']
ASN_KEEP = ['CB', 'HB2', 'HB3', 'CG', 'OD1', 'ND2', 'HD21', 'HD22']
ILE_KEEP = ['CA', 'HA', 'CB', 'HB', 'CG1', 'HG12', 'HG13',
            'CG2', 'HG21', 'HG22', 'HG23', 'CD1', 'HD11', 'HD12', 'HD13',
            'C', 'O', 'OXT']
# substrate: the bis-allylic donor and the two flanking vinyl carbons each side
SUBSTRATE_HEAVY = ['C14', 'C13', 'C17', 'C10', 'C15']

FRAGMENT_CHARGE = {'FE': +3, 'OH': -1, 'HIS': 0, 'ASN': 0, 'ILE': -1, 'LIG': 0}


def _classify(resname):
    n = resname.upper()
    if n.startswith('FE'):  return 'FE'
    if n.startswith('OH'):  return 'OH'
    if n.startswith('HD') or n == 'HIS' or n.startswith('HI'): return 'HIS'
    if n.startswith('ASN'): return 'ASN'
    if n.startswith('IE') or n == 'ILE': return 'ILE'
    if n.startswith('LIG'): return 'LIG'
    return None


def build_cluster(prm, elements, pos, donor_idx, xferH_idx,
                  substrate_heavy=SUBSTRATE_HEAVY, verbose=True):
    """Return (atoms, charge, mult, report).

    atoms  : list of (element, xyz, label); the transferring H is NOT included
             (the scanner inserts it at each grid point)
    charge : formal net charge from explicit per-fragment charges
    mult   : spin multiplicity consistent with electron parity
    report : dict with composition, caps added, and validation results
    """
    keep = []                      # prmtop indices retained
    frag_of = {}                   # idx -> fragment class
    donor_res = prm.atoms[donor_idx].residue.number

    for r in prm.residues:
        cls = _classify(r.name)
        if cls is None:
            continue
        if cls == 'LIG' and r.number != donor_res:
            continue
        if cls in ('FE', 'OH'):
            names = None                      # keep whole residue
        elif cls == 'HIS':
            names = HIS_KEEP
        elif cls == 'ASN':
            names = ASN_KEEP
        elif cls == 'ILE':
            names = ILE_KEEP
        elif cls == 'LIG':
            names = None                      # handled below by heavy list + H
        # Fe-proximity gate for the protein ligands (not for Fe/OH/substrate)
        if cls in ('HIS', 'ASN', 'ILE'):
            fe = [a.idx for rr in prm.residues if _classify(rr.name) == 'FE'
                  for a in rr.atoms]
            if fe:
                fpos = pos[fe[0]]
                if min(np.linalg.norm(pos[a.idx] - fpos) for a in r.atoms) > 3.5:
                    continue
        for a in r.atoms:
            if cls == 'LIG':
                if a.name in substrate_heavy:
                    keep.append(a.idx); frag_of[a.idx] = cls
                else:
                    # keep hydrogens bonded to a retained substrate heavy atom
                    if elements[a.idx] == 'H':
                        for b in a.bonds:
                            o = b.atom2 if b.atom1.idx == a.idx else b.atom1
                            if o.name in substrate_heavy and o.residue.number == r.number:
                                keep.append(a.idx); frag_of[a.idx] = cls
                                break
            elif names is None or a.name in names:
                keep.append(a.idx); frag_of[a.idx] = cls

    keep = sorted(set(keep))
    in_cluster = set(keep) | {xferH_idx}

    # --- cap every severed bond ------------------------------------------
    caps = []
    caps_on = defaultdict(int)          # prmtop idx -> number of caps attached
    for i in keep:
        if elements[i] == 'H' or elements[i] == 'Fe':
            continue
        a = prm.atoms[i]
        for b in a.bonds:
            o = b.atom2 if b.atom1.idx == i else b.atom1
            if o.idx in in_cluster:
                continue
            v = pos[o.idx] - pos[i]
            d = np.linalg.norm(v)
            if d < 1e-6:
                continue
            L = CAP_LENGTH.get(elements[i], 1.09)
            caps.append(('H', pos[i] + v / d * L,
                         f'cap:{a.residue.name}{a.residue.number}:{a.name}-{o.name}'))
            caps_on[i] += 1

    atoms = []
    for i in keep:
        if i == xferH_idx:
            continue
        a = prm.atoms[i]
        lbl = 'donor-C' if i == donor_idx else f'{a.residue.name}{a.residue.number}:{a.name}'
        atoms.append((elements[i], pos[i].copy(), lbl))
    atoms.extend(caps)

    # --- formal charge and parity ----------------------------------------
    seen_frag = set()
    charge = 0
    for i in keep:
        a = prm.atoms[i]
        key = (a.residue.name, a.residue.number)
        if key in seen_frag:
            continue
        seen_frag.add(key)
        charge += FRAGMENT_CHARGE[frag_of[i]]

    Zsum = sum(ATOMIC_NUMBER.get(e, 0) for e, _, _ in atoms) + 1   # +1 = xfer H
    n_e = Zsum - charge
    mult = 6                                   # Fe(III) high-spin d5, S = 5/2
    if (mult - 1) % 2 != n_e % 2:
        raise ValueError(
            f'Electron-parity failure after cluster construction: sum(Z)={Zsum}, '
            f'charge={charge:+d} -> {n_e} electrons '
            f'({"even" if n_e % 2 == 0 else "odd"}), incompatible with mult={mult}. '
            f'The fragment charge table or the atom selection is wrong.')

    # --- validation against the PARENT topology's own bond list -----------
    # A heavy atom is under-coordinated iff the number of its parent bonds is
    # not matched by (partners retained in the cluster) + (caps attached).
    # This is exact and hybridization-agnostic: it never guesses whether an
    # atom is sp2 or sp3, it only checks that no bond was silently dropped.
    bad = []
    for i in keep:
        if elements[i] in ('H', 'Fe'):
            continue
        a = prm.atoms[i]
        parent_bonds = len(a.bonds)
        retained = sum(1 for b in a.bonds
                       if (b.atom2 if b.atom1.idx == i else b.atom1).idx in in_cluster)
        if retained + caps_on[i] != parent_bonds:
            bad.append((f'{a.residue.name}{a.residue.number}:{a.name}', elements[i],
                        retained + caps_on[i], parent_bonds))

    comp = defaultdict(int)
    for e, _, _ in atoms:
        comp[e] += 1
    comp['H'] += 1                              # transferring H
    report = {
        'n_atoms': len(atoms) + 1,
        'n_caps': len(caps),
        'composition': dict(comp),
        'sum_Z': Zsum,
        'charge': charge,
        'n_electrons': n_e,
        'mult': mult,
        'S2_expected': ((mult - 1) / 2) * ((mult - 1) / 2 + 1),
        'undercoordinated': bad,
    }
    if verbose:
        print(f'  cluster: {report["n_atoms"]} atoms '
              f'({"".join(f"{e}{n}" for e, n in sorted(comp.items()))}), '
              f'{len(caps)} H caps')
        print(f'  charge {charge:+d}, {n_e} electrons, mult {mult}, '
              f'<S^2> expected {report["S2_expected"]:.2f}')
    if bad:
        raise ValueError(f'Cluster validation FAILED: {len(bad)} under-coordinated '
                         f'heavy atoms, e.g. {bad[:5]}')
    return atoms, charge, mult, report

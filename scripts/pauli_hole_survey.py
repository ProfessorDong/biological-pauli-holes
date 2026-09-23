"""Survey of Pauli-hole radii around hydrogen across biological and reference sites.

For each site with a defined hydrogen position we measure the cavity radius xi0 and
evaluate the You-Ye half-confinement solution (scripts/pauli_hole_hydrogen.py) to obtain
the induced dipole P, the ground-state energy shift, and the axial dipole field.

DEFINITION OF xi0 (stated once, applied uniformly)
  xi0 = distance from the hydrogen to the nearest NON-BONDED closed-shell heavy atom.
  "Closed-shell heavy atom" means C, N, O, S, P, Se, Cl.
  "Non-bonded" excludes any atom that is a covalent or dative partner of the hydrogen,
  operationally any heavy atom closer than BOND_CUT, and excludes ALL open-shell metals
  at any distance. Both the bonded and non-bonded nearest neighbours are reported so the
  exclusion is auditable.

DOMAIN OF VALIDITY (recorded per site, not hidden)
  The model describes a hydrogen only weakly held in a closed-shell cavity; the original
  work states explicitly that it does not describe a chemisorbed/bonded hydrogen. Each
  site is therefore flagged:
    'free'   - no heavy atom within BOND_CUT and no metal within METAL_CUT: model applies
    'bonded' - H is covalently bound to a heavy atom: model is an approximation only
    'metal'  - H is bound to one or more open-shell metals: model is OUT OF DOMAIN
  Sites flagged 'metal' are reported for geometric context and must NOT be read as
  quantitative applications of the model.

Usage: pauli_hole_survey.py
"""
import json, sys, urllib.request
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pauli_hole_hydrogen import solve, A0_ANG

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
OUT = ROOT / 'results' / 'pauli_hole_survey'
CACHE = OUT / 'pdb'
CLOSED_SHELL = {'C', 'N', 'O', 'S', 'P', 'SE', 'CL', 'F'}
METALS = {'NI', 'FE', 'MN', 'MG', 'ZN', 'CU', 'CO', 'MO', 'W', 'V', 'CA', 'NA', 'K'}
BOND_CUT = 1.35      # A; H covalently bound to C/N/O/S lies below this
METAL_CUT = 2.20     # A; metal-hydride bonds lie below this
K_COUL = 8.9875517873681764e9
EA0 = 8.4783e-30
R_FIELD = 2.55e-10   # evaluate the axial field at the SLO reactive separation
KSI_FIELD = 150.0    # MV/cm, ketosteroid isomerase reference


def fetch(pdb):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f'{pdb}.pdb'
    if not f.exists():
        url = f'https://files.rcsb.org/download/{pdb}.pdb'
        f.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return f


def read_atoms(path):
    out = []
    for L in open(path):
        if L.startswith(('ATOM', 'HETATM')):
            el = (L[76:78].strip() or L[12:16].strip()[0]).upper()
            out.append(dict(name=L[12:16].strip(), res=L[17:20].strip(), ch=L[21],
                            seq=L[22:26].strip(), el=el, alt=L[16],
                            xyz=np.array([float(L[30:38]), float(L[38:46]), float(L[46:54])])))
    return [a for a in out if a['alt'] in (' ', 'A')]


def analyse(H, atoms, label, note=''):
    """Measure the cavity around hydrogen position H and evaluate the model."""
    others = [a for a in atoms if np.linalg.norm(a['xyz'] - H) > 1e-6]
    d = np.array([np.linalg.norm(a['xyz'] - H) for a in others])
    order = np.argsort(d)

    bonded_heavy, bonded_metal, nn_free = [], [], None
    for i in order:
        a, dist = others[i], d[i]
        if dist > 8.0:
            break
        if a['el'] in METALS:
            if dist < METAL_CUT:
                bonded_metal.append((dist, a))
            continue
        if a['el'] == 'H':
            continue
        if a['el'] not in CLOSED_SHELL:
            continue
        if dist < BOND_CUT:
            bonded_heavy.append((dist, a))
            continue
        if nn_free is None:
            nn_free = (dist, a)

    flag = 'metal' if bonded_metal else ('bonded' if bonded_heavy else 'free')
    if nn_free is None:
        return None
    xi_A = nn_free[0]
    x = xi_A / A0_ANG
    r = solve(x)
    E = K_COUL * 2 * r['P'] * EA0 / R_FIELD**3 / 1e8
    return dict(label=label, note=note, flag=flag,
                xi0_A=xi_A, xi0_over_a0=x,
                nearest_free=f"{nn_free[1]['res']}{nn_free[1]['seq']} {nn_free[1]['name']}",
                bonded_heavy=[f"{a['res']}{a['seq']} {a['name']} {dd:.2f}A"
                              for dd, a in bonded_heavy[:3]],
                bonded_metal=[f"{a['res']}{a['seq']} {a['name']} {dd:.2f}A"
                              for dd, a in bonded_metal[:3]],
                P_ea0=r['P'], I_eV=r['I_eV'], dE_eV=13.605693122994 - r['I_eV'],
                E_MVcm=E, pct_KSI=100*E/KSI_FIELD)


def site_from_pdb(pdb, selector, label, note=''):
    atoms = read_atoms(fetch(pdb))
    H = selector(atoms)
    if H is None:
        print(f'  !! {label}: hydrogen not found in {pdb}')
        return None
    return analyse(H, atoms, label, note)


def pick(atoms, res=None, name=None, seq=None):
    for a in atoms:
        if res and a['res'] != res: continue
        if name and a['name'] != name: continue
        if seq and a['seq'] != str(seq): continue
        return a['xyz']
    return None


def slo_sites():
    """Soybean lipoxygenase: transferring H, from our own sampled MD frames."""
    import json as _j
    EF = ROOT / 'results' / 'ensemble_fluctuation'
    rows = []
    for tag in ['WT', 'I553A', 'I552A', 'L546A', 'L754A', 'V750A', 'I538A']:
        for clamp, cname in [('r255', 'reactive'), ('r340', 'reference')]:
            f = EF / f'{tag}_{clamp}_frames_geometry.json'
            if not f.exists():
                continue
            g = _j.loads(f.read_text())
            ds = []
            for fr in g['frames']:
                xH = np.array(fr['xferH'])
                best = min(np.linalg.norm(np.array(a['position']) - xH)
                           for a in fr['native_wall'] if a['element'] != 'H')
                ds.append(best)
            xi = float(np.mean(ds))
            x = xi / A0_ANG
            r = solve(x)
            E = K_COUL * 2 * r['P'] * EA0 / R_FIELD**3 / 1e8
            rows.append(dict(label=f'SLO {tag} ({cname})', note='transferring H, MD ensemble',
                             flag='bonded', xi0_A=xi, xi0_over_a0=x,
                             nearest_free='side-chain wall', bonded_heavy=['donor C ~1.09A'],
                             bonded_metal=[], P_ea0=r['P'], I_eV=r['I_eV'],
                             dE_eV=13.605693122994 - r['I_eV'],
                             E_MVcm=E, pct_KSI=100*E/KSI_FIELD))
    return rows


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    print('=== metal hydride cages (model OUT OF DOMAIN, geometric context only) ===')
    for pdb, note in [('4U9H', 'Ni-R(UH) subatomic X-ray, bridging hydride'),
                      ('4U9I', 'Ni-R(H) subatomic X-ray, bridging hydride')]:
        r = site_from_pdb(pdb, lambda A: pick(A, res='NWN', name='H'),
                          f'[NiFe] hydrogenase {pdb}', note)
        if r: results.append(r); print(f"  {r['label']:34s} xi0={r['xi0_A']:.2f} A  flag={r['flag']}")

    print('\n=== enzymatic transferring hydrogen (side-chain pocket) ===')
    for r in slo_sites():
        results.append(r); print(f"  {r['label']:34s} xi0={r['xi0_A']:.2f} A  flag={r['flag']}")

    print('\n=== reference: close-packed solid surface (You & Ye regime) ===')
    for lab, xi in [('Pd(110) rhomboid hole', 1.37), ('NaCl square hole', 1.85)]:
        x = xi / A0_ANG; rr = solve(x)
        E = K_COUL*2*rr['P']*EA0/R_FIELD**3/1e8
        results.append(dict(label=lab, note='surface Pauli hole', flag='free',
                            xi0_A=xi, xi0_over_a0=x, nearest_free='lattice',
                            bonded_heavy=[], bonded_metal=[], P_ea0=rr['P'],
                            I_eV=rr['I_eV'], dE_eV=13.605693122994-rr['I_eV'],
                            E_MVcm=E, pct_KSI=100*E/KSI_FIELD))
        print(f"  {lab:34s} xi0={xi:.2f} A  flag=free")

    (OUT / 'survey.json').write_text(json.dumps(results, indent=1))

    print('\n' + '='*96)
    print(f"{'site':34s}{'flag':8s}{'xi0(A)':>8}{'xi0/a0':>8}{'P(ea0)':>9}"
          f"{'dE(eV)':>8}{'E(MV/cm)':>10}{'%KSI':>7}")
    print('='*96)
    for r in sorted(results, key=lambda r: r['xi0_over_a0']):
        print(f"{r['label']:34s}{r['flag']:8s}{r['xi0_A']:>8.2f}{r['xi0_over_a0']:>8.2f}"
              f"{r['P_ea0']:>9.3f}{r['dE_eV']:>8.2f}{r['E_MVcm']:>10.1f}{r['pct_KSI']:>7.1f}")
    print(f"\nwrote {OUT/'survey.json'}")

#!/usr/bin/env python3
"""Check that the article, its appendices and the supplement say the same thing.

WHY THIS EXISTS
  Three documents carry overlapping accounts of the same calculations, and they have drifted
  apart twice. After the V750A repair the appendix kept every superseded endpoint while the main
  text was corrected. After the rate partition was withdrawn in the appendix, the supplement went
  on reporting a semiclassical KIE of 7.9 and a tunneling amplification of 8.4. Each time a build
  was clean and every reference resolved, so compilation success proved nothing.

  The checks below are the specific contradictions that actually occurred, plus the claims an
  audit found stated in one place and denied in another. Each is a substring test on the SOURCE,
  which is enough because the failure mode is a claim surviving in one file after being removed
  from another.

Usage: verify_document_consistency.py     (pauli env; exit status is the failure count)
"""
import re
import sys
from pathlib import Path

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/prxlife')
MAIN, APP = (ROOT / 'main.tex'), (ROOT / 'appendices.tex')
# The separate Supplemental Material was retired on 2026-09-18 and folded into the
# appendices, which removes the drift vector these checks were written for. The
# checks are kept and re-pointed, because the claims themselves must still not return.
SUP = APP
COVER = ROOT / 'cover-letter.tex'

# (description, file, text, must_be_present)
CHECKS = [
    # the withdrawn reactant-ZPE rate partition must not survive anywhere
    ('rate partition withdrawn in supplement', SUP, 'we withdraw it', True),
    ('no tunneling amplification claimed in supplement', SUP,
     'is the additional amplification attributable to tunneling', False),
    ('no tunneling amplification claimed in appendix', APP,
     'is the tunneling amplification', False),
    # an adiabatic state can change character; only the followed branch was at fault
    ('supplement does not assert adiabatic impossibility', SUP,
     'cannot relax the electronic structure', False),
    ('main text does not assert adiabatic impossibility', MAIN,
     'cannot relax the electronic structure', False),
    # the separation curvature must not be converted into a nuclear frequency
    ('appendix withdraws the ZPE conversion', APP, 'That conversion is withdrawn', True),
    ('main text does not call k_exch a transverse curvature', MAIN,
     'transverse exchange-repulsion curvature $k_\\mathrm{exch}', False),
    # coordinate invariance of a covariance determinant
    ('supplement does not claim coordinate invariance for det Sigma', SUP,
     'coordinate-invariant joint 2D reactive-basin volume', False),
    # the force field: two cluster models, and AM1-BCC for the substrate
    ('appendix states AM1-BCC for the substrate', APP, 'AM1-BCC', True),
    ('appendix does not claim sub-0.05 A crystal agreement', APP,
     'agreement to $<0.05$~\\AA\\ on all six coordinating', False),
    # proposed protocols must be marked
    # the unexecuted protocols moved to the appendices in the 2026-09-18 restructure
    ('unexecuted protocols are marked as such', APP,
     'Proposed protocol, not carried out here', True),
    ('Bondi caution recorded with the radii', APP,
     'may not always be suitable for the calculation of contact distances', True),
    ('contact distance no longer asserted as a floor', APP,
     'cannot approach the hydrogen closer than', False),
    ('bead count stated as below the criterion', MAIN,
     'The bead count is below the $P\\approx32$', True),
    ('superseded bead rule of thumb withdrawn', MAIN, 'suggests $P\\!\\gtrsim\\!14$', False),
    ('cover letter drops the contact-distance floor', COVER, 'beyond $2.90', False),
    # the compressed ensemble is not a transfer geometry
    ('main text does not claim a transfer geometry', MAIN,
     'where hydrogen transfer actually occurs', False),
    ('main text does not claim only the wall differs', MAIN, 'so that only the cavity wall differs', False),
    # the sensitivity calculation is not a bound
    ('main text calls the ZPE channel a sensitivity calculation', MAIN,
     'a sensitivity calculation and not a bound', True),
    # --- revision-2 audit: claims described as withdrawn that had survived in print ---
    ('no biological index bound in the appendix', APP, 'confinement index therefore satisfies', False),
    ('no accessible band in the decisive-figure caption', APP,
     'shaded band marks the range a protein can present', False),
    ('water benchmark claims no isotope scaling', APP, 'carries the exact isotope scaling', False),
    ('water benchmark does not lead with ZPE', APP, 'Zero-point energies follow', False),
    ('benchmark heading drops isotope-sensitive', APP,
     'well defined and isotope-sensitive', False),
    ('no categorical proxy claim in main text', MAIN,
     'does not preserve rank is not a proxy', False),
    ('main text uses fixed longitudinal separation, not bare separation', MAIN,
     'fixed \\emph{longitudinal} separation', True),
    ('no exclusion inferred from the null endpoints', APP,
     'exclude is accordingly an association close to deterministic', False),
    ('leverage dismissal withdrawn', APP, 'leverage that seven points cannot absorb', False),
    ('no transfer-competence claim in the appendix', APP,
     'where the reaction does not occur', False),
    ('packing scenario not stated as a respected bound', APP,
     'Every value we measure falls at or below that bound', False),
    ('below-kT claim withdrawn', APP,
     'whole effect lies at or below the thermal energy', False),
    ('index calibration marked prospective', APP, 'are calibrated from protein data', False),
    ('gating appendix claims no rare-configuration rate', APP,
     'the apparent enzyme rate is a rare-configuration rate', False),
    ('stale field percentage removed', APP, '(52\\%)', False),
    ('superseded achieved range removed', MAIN, '2.775$ to $2.809', False),
    ('superseded residual spread removed', MAIN, '0.065\\,\\si{\\angstrom}$ residual spread', False),
    # --- author instruction: no mention of AI anywhere in the submission ---
    ('no AI mention in main text', MAIN, 'AI', False),
    ('no AI mention in appendices', APP, 'AI', False),
    ('no AI mention in cover letter', COVER, 'AI', False),
    ('no model vendor named in main text', MAIN, 'Anthropic', False),
    ('no model vendor named in appendices', APP, 'Anthropic', False),
    ('no assistant named in main text', MAIN, 'Claude', False),
    ('no assistant named in appendices', APP, 'Claude', False),
    ('no assistant named in cover letter', COVER, 'Claude', False),
    ('no "generated with" phrasing in appendices', APP, 'generated with', False),
    ('no language-model mention in main text', MAIN, 'language model', False),
    ('no language-model mention in appendices', APP, 'language model', False),
    # title agreement across documents
    # the fold itself: the RPMD account and all four former supplement figures must now be
    # inside the article, and no stray Supplemental Material pointer may remain in the text
    ('RPMD account folded into the appendices', APP,
     'Equilibrium quantum delocalization on the I553A/I552A pair', True),
    ('index figure folded in', APP, 'fig2_index.pdf', True),
    # fig4_dispersion was REMOVED 2026-09-23. It plotted a withdrawn correlation, and a
    # figure is quoted more easily than the caption that retracts it. The appendix
    # section is retained in full, so the record survives without the artwork.
    ('withdrawn dispersion figure is not reinstated', APP, 'fig4_dispersion.pdf', False),
    ('density figure folded in', APP, 'fig6_pimd_density.pdf', True),
    ('convergence figure folded in', APP, 'figED2_bead_convergence.pdf', True),
    ('no Supplemental Material pointers left', MAIN, 'Supplemental Material Sec', False),
    ('cover letter carries the current title', COVER,
     'An exponential bound on Pauli confinement', True),
    ('cover letter drops the framing-A title', COVER,
     'Fragment-separation curvature is not', False),
    ('no document keeps the old title', SUP, 'locality of direct exchange', False),
    ('main text carries the current title', MAIN, '\\title{' + "An exponential bound on Pauli confinement in enzyme active sites" + '}', True),
    ('main text drops the framing-A title', MAIN, "Fragment-separation curvature is not hydrogen confinement", False),
    ('no document keeps the superseded title', MAIN,
     'Testing hydrogen confinement by exchange', False),
    ('no document keeps the revision-3 title', MAIN,
     'Distinguishing exchange repulsion from hydrogen', False),
    ('no document keeps the colon title', MAIN,
     'Moving the wall is not moving the hydrogen', False),
    ('cover letter drops the superseded title', COVER,
     'Testing hydrogen confinement by exchange', False),
    ('cover letter drops the old title', COVER, 'locality of direct exchange', False),
    # superseded budget numbers
    ('cover letter uses the corrected budget median', COVER, '0.13\\%', False),
    ('cover letter drops the superseded isotope factor', COVER, '1.16', False),
    # ---- stage 5, the 101-configuration thermal ensemble of K_perp (added 2026-09-22) ----
    # The ensemble closed a gap main.tex had DECLARED in writing, so the old wording must not
    # return, and the one static claim it overturned must stay qualified.
    ('main text no longer calls the transverse Hessian un-averaged', MAIN,
     'has no ensemble averaging', False),
    ('main text reports the thermal ensemble', MAIN,
     'No system changes class', True),
    ('abstract carries the ensemble result', MAIN,
     '$101$-configuration thermal\nensemble', True),
    ('body carries the ensemble result', MAIN, 'No system changes class', True),
    # K_perp^int was negative definite in six of seven ONLY at the static geometries.
    # Over the ensemble it is five. The qualifier is the whole point and must survive.
    ('total interaction six-of-seven is qualified as geometry-specific in the main text', MAIN,
     'six of the seven systems on these configurations', True),
    ('body qualifies the total-interaction count as geometry-specific', MAIN,
     'six of the seven systems on these configurations', True),
    ('body records the ensemble count as five', MAIN,
     'over the thermal ensemble it is five', True),
    ('appendix records that the ensemble count is five, not six', APP,
     'the count is five', True),
    ('appendix does not present six-of-seven as an ensemble property', APP,
     'negative definite in six of seven over the ensemble', False),
    # the pre-registered plan was deviated from in three ways; all must stay disclosed
    ('appendix discloses the deviations from the declared stage-5 plan', APP,
     'three deviations from the declared plan', True),
    ('appendix reports the declared median statistic', APP,
     'negative at the median configuration', True),
    # V750A is ambiguous in BOTH analyses and is counted in neither column
    ('appendix keeps V750A out of both columns after the ensemble', APP,
     'We continue to count V750A in\nneither column', True),
    # the discarded first segment is a protocol fact, not an outcome-dependent choice
    ('appendix states the seg00 discard rule was applied before inspection', APP,
     'applied before the transverse Hessians were\ninspected', True),
    # r_DA ensembles must not be conflated
    ('appendix states the stage-5 achieved r_DA range', APP,
     '$2.744$ to $2.786', True),
    ('appendix warns the two r_DA ensembles are not interchangeable', APP,
     'should not be quoted against', True),

    # ---- stage 4, numerical and model sensitivity (added 2026-09-23) ----
    # The sign survives all six settings; the donor fragment sets a 28% magnitude uncertainty.
    ('main text no longer calls SAPT-order sensitivity an open refinement', MAIN,
     'ALMO-EDA sensitivity remains the natural next-round refinement', False),
    ('main text reports the six-setting sweep', MAIN,
     'The sign of the smaller eigenvalue survives all six', True),
    ('main text attaches the fragment uncertainty to magnitudes', MAIN,
     'fragment-definition uncertainty of about $28\%$', True),
    ('appendix states the sign is invariant across settings', APP,
     'Nothing tested reverses the classification', True),
    ('appendix names the donor fragment as the dominant sensitivity', APP,
     'the donor fragment at $27.6\%$', True),
    ('appendix reports SAPT2+ preserving the sign', APP,
     'leaves its sign and the\nwhole argument intact', True),
    ('appendix records the wall truncation as not load-bearing', APP,
     'is not\nload-bearing', True),
    ('appendix keeps ALMO-EDA declared as unexecuted', APP,
     'ALMO-EDA on the native pair was not part of this sweep', True),
    ('appendix reports the gradient so the reader can check it', APP,
     'separation working rather than failing', True),

    # ---- framing B: the exponential bound leads the paper (2026-09-23) ----
    ('abstract leads with the exponential bound', MAIN,
     'question about a single distance', True),
    ('abstract states the quantitative exclusion', MAIN,
     'Confinement is therefore not the\ncatalytic field', True),
    ('abstract keeps the measured-not-assumed discipline', MAIN,
     'We measure that distance rather than assume it', True),
    ('exponential section precedes the coordinate section', MAIN,
     'How far direct exchange reaches', True),
    ('discussion opens on the quantitative answer', MAIN,
     'The question has a quantitative answer', True),
    ('discussion no longer calls fragment sensitivity outstanding', MAIN,
     'whether they survive changes of fragment size and electronic treatment,\nis what is left', False),
    ('cover letter argues the biological-physics case, not a methods case', COVER,
     'answers a mechanistic question in biological physics', True),

    # The figure captions were not in the sweep that qualified this claim, and the
    # Figure 5 caption carried the bare form while abstract, Results and appendix all
    # carried the qualified one. A present-check could not catch that; this absent-check can.
    ('no bare six-of-seven total-interaction claim survives in the main text', MAIN,
     'negative definite in six of seven systems', False),
    ('figure 5 caption qualifies the total-interaction count', MAIN,
     'six of the seven systems \\emph{at these geometries}', True),

    # The Figure 8 caption attributed R = +0.15 to an "MBAR-reweighted MEAN r_DA".
    # The appendix section it summarises is titled "MBAR reanalysis of sigma_{r_DA}",
    # and the withdrawn descriptor is built from standard deviations, not means.
    ('retained section names sigma, not a mean, for the MBAR recomputation', APP,
     'MBAR reanalysis of $\\sigma_{r_{\\mathrm{DA}}}$', True),
    ('the removal is recorded rather than silent', APP,
     'An earlier version of this appendix carried a two-panel figure', True),
    ('no MBAR-reweighted mean r_DA is claimed', APP,
     'MBAR-reweighted mean $r_\\mathrm{DA}$', False),

    # This campaign uses a NATIVE WALL with a CONSTRUCTED METHANE DONOR
    # (sapt_native_fragment.py: Hs_donor = build_methane). The caption used to say the
    # values were "not from the methane surrogates", which reads as neither being one.
    ('figED1 caption admits the donor is still a methane surrogate', APP,
     'the donor is\nstill a constructed methane', True),
    ('figED1 caption does not claim freedom from methane surrogates', APP,
     'not from the methane surrogates', False),

    # Ley-Koo and Garcia-Castelan (1991) solve the SAME single open paraboloid and tabulate
    # xi0 = 10.01 for the dipole (inside our 9.1 to 11.3 biological range) and xi0 = 11.5565
    # for the boundary root (above it).  The abstract claimed 9.1 to 11.3 lay "a factor of two
    # to seven beyond the largest tabulated solution of this problem", which is false on both
    # counts: the factor beyond You and Ye's largest entry is 1.9 to 2.4, and this problem IS
    # tabulated in that range by a paper our own validation script consumes.  The claim was
    # written before that paper was found and outlived the appendix's correction of it.
    ('abstract does not claim the biological range is beyond all tabulations', MAIN,
     'beyond the largest tabulated solution', False),
    ('abstract does not inflate the You and Ye extrapolation factor', MAIN,
     'a factor of two to seven', False),
    ('appendix still discloses the Ley-Koo dipole disagreement by row', APP,
     '$P=0.0041$ at $\\xi_0=10.01$', True),
    ('appendix still credits Ley-Koo and Garcia-Castelan for the same boundary', APP,
     'The same single open paraboloidal boundary was also solved directly', True),

    # Two different matrices give two different counts and they must not be swapped.
    # K_perp^exch: smaller eigenvalue negative in 6 of 7 ensemble means (all but L754A), over
    # 16+5+16+16+16+16+16 = 101 configurations.  K_perp^int: negative DEFINITE in 6 of 7 at the
    # static geometries but only 5 of 7 over the ensemble, V750A's larger eigenvalue averaging
    # +0.0091 N/m.  Verified against stage5_ensemble_panel.json on 2026-09-23.
    ('ensemble negative-definite count is five, not six', MAIN,
     'over the thermal ensemble it is five of seven', True),
    ('static negative-definite count carries the geometry qualifier', MAIN,
     'negative definite in six of the seven systems \\emph{at these geometries}', True),
    ('the anti-confining eigenvalue count is not left unqualified', APP,
     'anti-confining eigenvalue we find in six of seven systems, a count', True),

    # Verified against references/1803.01037.pdf on 2026-09-23.  The source says "a typical
    # system containing O-H covalent bonds", not X-H, and it does NOT state a
    # P ~ hbar*w_max/kT criterion anywhere; that inference is ours.  Both had been folded into
    # a single \\cite{markland2018}.  O-H is the higher frequency, so their P=32 overstates
    # what our C-H bond needs, which makes the self-criticism conservative rather than inflated.
    ('the P=32 requirement is attributed to O-H as the source states', MAIN,
     'simple structural properties of O--H bonds at room temperature requires', True),
    ('the P=32 requirement is not generalised to X-H', MAIN,
     'structural properties of X--H bonds', False),
    ('the bead-count criterion is marked as our inference, not the source\'s', APP,
     'it is ours rather than theirs', False if False else True),
    ('the criterion is not presented as the accepted one', APP,
     'The accepted criterion is that $P$ must be a low multiple', False),

    # Verified against references/ar500322s.pdf on 2026-09-23.  The quotation is verbatim (only
    # sentence-initial "From" lowercased for embedding) and "orthogonal" occurs exactly ONCE in
    # the whole paper, in a concluding summary, so "asserted qualitatively without an associated
    # observable" is exact.  But the passage had said we make his stiffening picture quantitative,
    # when our intermolecular transverse curvature is anti-confining in six of seven: opposite
    # sign.  Claiming to quantify an antecedent we actually reverse both misdescribes it and
    # undersells the result.
    ('the Kohen antecedent is reported as reversed, not confirmed', APP,
     'Making it quantitative reverses its sign', True),
    ('we do not claim merely to be quantifying his stiffening picture', APP,
     'transverse-stiffening picture this paper tries to make', False),

    # The measured closest approaches are d = 2.42 to 2.95 A (appendices: reactive clamp
    # 2.42-2.86, reference clamp 2.71-2.95).  Rounding those OUTWARD to 2.40 and 3.00 before
    # exponentiating inflated the whole chain: xi0/a0 9.1-11.3 instead of 9.1-11.1, P
    # 0.0043-0.021 instead of 0.0049-0.020, field 0.4-1.9 instead of 0.45-1.81.  tab:expbound
    # always had the correct values; the abstract, Results, a figure caption, an appendix
    # paragraph AND the fig3 artwork carried the inflated ones.  Fixed 2026-09-23.
    ('the measured closest approach is not rounded outward', MAIN,
     '$d=2.42$ to $2.95', True),
    ('the confinement range is not inflated to 11.3', MAIN,
     '9.1$ to $11.3', False),
    ('the dipole range matches the exact solution at the measured distances', MAIN,
     '0.0049$ to\n$0.020', True),
    ('the inflated dipole lower bound is gone', MAIN,
     '0.0043', False),
    ('the appendix dipole range is not inflated', APP,
     '0.021\\,ea_0', False),

    # ---- table audit, 2026-09-23.  Each verified against a named field of a named file.
    # tab:budget: the >8 A column was zero BY CONSTRUCTION (confinement_budget.py truncated the
    # atom list at 8 A) and the true tail is 0.7-1.7e-9, so "<1e-9" was false for 9 of 14.
    ('the beyond-8A tail is not claimed below 1e-9', MAIN,
     '$95.1$--$98.9$ & $<10^{-9}$', False),
    ('the beyond-8A tail is stated over every atom', MAIN,
     'evaluated over every\natom of the system and not over any truncated neighbour list', True),
    # tab:kie: the lower block is 30 C external data and cannot come from Hu 2019 (10/40 C).
    ('the KIE table does not claim all values come from Hu 2019', APP,
     'All values are transcribed from Table 1 of', False),
    ('the external KIE block carries its own sources', APP,
     'the Ile553 series from the experimental rows of Table 1 of Meyer and Klinman', True),
    # tab:youye-validation: 7 of 10 dipoles are within 0.3%, not 9; 8 of 10 radii within 2%, not 9.
    ('the You-Ye dipole threshold is 0.4 percent', MAIN,
     'nine of the ten tabulated rows within\n$0.4\\%$', True),
    ('the You-Ye radius threshold is not 2 percent', APP,
     'nine of ten within $2\\%$', False),
    # tab:ensfluct: the true displacement range is 0.15 to 1.20 sd, not 0.4 to 1.2.
    ('the ensemble displacement range is not understated', APP,
     'displaced from these means by $0.4$ to $1.2$ standard deviations', False),
    # tab:qmpes: the data are the 15-point production run, NOT the invalidated pcet_pilot.
    ('the QM scan table is not called a pilot', APP,
     'results of the WT proton-scan pilot', False),

    # ---- equation audit, 2026-09-23.
    # eq:Kperp and eq:curv defined the curvature at a transverse MINIMUM with no linear term,
    # while the estimator expands at the hydrogen's sampled position, which the appendix itself
    # shows is not stationary (linear term ~9x the quadratic), and fits and tabulates that
    # gradient.  Both now retain g and expand at q_perp = 0.
    ('eq:Kperp retains the linear term', APP,
     '\\mathbf g(\\mathbf R)^{\\!\\top}\\mathbf q_\\perp', True),
    ('no curvature is defined at a transverse minimum', APP,
     'about its transverse minimum', False),
    ('eq:curv is not expanded about the transverse minimum', APP,
     'order about the transverse minimum', False),
    # eq:zpe wrote sqrt(k_exch/m) two lines after its own lead-in wrote sqrt(K_perp/m).
    # k_exch^bio is K_sep; assigning it a frequency is the retracted error.
    ('eq:zpe uses K_perp and not k_exch', APP,
     '\\tfrac12\\hbar\\sqrt{K_\\perp/m_i}', True),
    # tab:kexch called a fragment-separation curvature "Transverse".
    ('tab:kexch is not called a transverse curvature', APP,
     'Transverse exchange-repulsion curvature $k_{\\mathrm{exch}}^{\\mathrm{bio}}$', False),
    ('tab:kexch names the separation coordinate', APP,
     'Fragment-separation exchange-repulsion curvature', True),
    # the retracted energy/curvature equivalence survived under eq:pi
    ('the energy is not equated with the curvature at eq:pi', APP,
     'equivalently, the transverse curvature of $V_{\\mathrm{Pauli}}$', False),
    # main.tex Eq. 3 is the Hamiltonian; the confinement index is not in the main text at all
    ('the confinement index is not cited as main-text Eq. 3', APP,
     'Eq.~3 of the main text', False),
    ('no hard-coded equation number survives', APP,
     'of Eq.~(2) sharpens', False),

]


def main():
    """Whitespace-normalized substring tests.

    An audit found that line breaks in the LaTeX source defeated literal matching: a claim
    could be removed from one document, survive in another, and the checker would report
    both as clean because the wrapped text did not match. Every needle and haystack is
    therefore collapsed to single spaces before testing.
    """
    def flat(x):
        return re.sub(r'\s+', ' ', x)

    cache = {}
    bad = 0
    width = max(len(c[0]) for c in CHECKS)
    for desc, f, text, want in CHECKS:
        if f not in cache:
            cache[f] = flat(f.read_text())
        present = flat(text) in cache[f]
        ok = (present == want)
        bad += not ok
        print(f'  [{"OK " if ok else "FAIL"}] {desc:<{width}}  ({f.name}: '
              f'{"present" if present else "absent"})')
    print(f'\n  {len(CHECKS)-bad}/{len(CHECKS)} consistency checks passed')
    return bad


if __name__ == '__main__':
    sys.exit(main())

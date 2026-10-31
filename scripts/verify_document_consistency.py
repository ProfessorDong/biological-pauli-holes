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
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import re
import sys
from pathlib import Path

ROOT = Path(_REPO + '/prxlife')
MAIN, APP = (ROOT / 'main.tex'), (ROOT / 'appendices.tex')
# The separate Supplemental Material was retired on 2026-09-18 and folded into the
# appendices, which removes the drift vector these checks were written for. The
# checks are kept and re-pointed, because the claims themselves must still not return.
SUP = APP
COVER = ROOT / 'cover-letter.tex'

# (description, file, text, must_be_present)
CHECKS = [
    # the withdrawn reactant-ZPE rate partition must not survive anywhere
    ('no isotope scaling is formed from the O...O curvature', SUP,
     'An isotope scaling obtained by inserting a chosen mass into a curvature measured along a different coordinate is arithmetic, not evidence', True),
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
    ('appendix forms no ZPE from the separation curvature', APP,
     'so it supports no frequency and no zero-point energy', True),
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
    # 2026-10-31 float audit: the index figure duplicated fig:coordinate (two coordinates) and
    # illustrated an index the paper never evaluates; it was removed, not folded.
    ('index figure removed as redundant', APP, 'fig2_index.pdf', False),
    # every float is cited from the text; these eleven had no \ref before the audit
    ('fig:coordinate is cited', MAIN, 'Fig.~\\ref{fig:coordinate}', True),
    ('tab:wallcheck is cited', APP, 'Table~\\ref{tab:wallcheck}', True),
    ('tab:scalarsummary is cited', APP, 'Table~\\ref{tab:scalarsummary}', True),
    ('tab:transfergeom is cited', APP, 'Table~\\ref{tab:transfergeom}', True),
    ('tab:qmpes is cited', APP, 'Table~\\ref{tab:qmpes}', True),
    ('tab:fc_gating is cited', APP, 'Table~\\ref{tab:fc_gating}', True),
    ('tab:ensfluct is cited', APP, 'Table~\\ref{tab:ensfluct}', True),
    ('tab:expbound is cited', APP, 'Table~\\ref{tab:expbound}', True),
    ('fig:fields is cited', APP, 'Fig.~\\ref{fig:fields}', True),
    ('fig:decisive is cited', APP, 'Fig.~\\ref{fig:decisive}', True),
    ('fig:pimd is cited from its own appendix', APP, '(Fig.~\\ref{fig:pimd})', True),
    # one definition of the transverse curvature (eq:curv) and one rate kernel (eq:HSrate)
    ('the transverse curvature is defined once', APP, 'label{eq:Kperp}', False),
    ('the rate kernel is displayed once', APP, 'label{eq:kchannels}', False),
    ('the dipole asymptote is displayed once', APP, 'label{eq:expbound-si}', False),
    ('no stale pointer to a Methods subsection in tab:kexch', APP,
     'SAPT exchange curvature at biological geometry', False),
    ('no clamp is called where transfer occurs', APP, 'at the separations where transfer occurs', False),
    # fig4_dispersion was REMOVED 2026-09-23. It plotted a withdrawn correlation, and a
    # figure is quoted more easily than the caption that retracts it. The appendix
    # section is retained in full, so the record survives without the artwork.
    ('withdrawn dispersion figure is not reinstated', APP, 'fig4_dispersion.pdf', False),
    # 2026-10-31: this is a first submission. A reader has no "earlier version" to compare
    # against, so the manuscript states the correct result and carries no correction diary.
    # The withdrawn claims are still guarded by the absence checks throughout this file.
    ('main text carries no earlier-version narrative', MAIN, 'earlier version', False),
    ('main text carries no earlier-draft narrative', MAIN, 'earlier draft', False),
    ('appendix carries no earlier-version narrative', APP, 'earlier version', False),
    ('appendix carries no retraction subsection', APP, 'app:retraction', False),
    ('appendix does not call the exploratory descriptor withdrawn in a heading', APP,
     'campaigns and a withdrawn descriptor', False),
    # 2026-10-31: three review PDFs (hs2015, fried2017, klinman2013) read in full; the sentences
    # they now anchor were checked against the printed text, and a sentence that rendered as
    # "different derivatives Second," was repaired.
    ('I553 comparison sentence is complete', MAIN, 'different derivatives Second', False),
    ('KIE temperature dependence is tied to DAD sampling (Klinman & Kohen 2013)', MAIN,
     'their temperature dependence reports how much the donor--acceptor distance must be sampled', True),
    ('classical force-field resolution caveat (Hammes-Schiffer 2015) is stated', MAIN,
     'classical force fields often cannot resolve', True),
    ('the two clamps are bracketed against Klinman & Kohen 2013', MAIN,
     'The pair brackets the separations Klinman and Kohen contrast', True),
    ('KSI field attributed to short polar contacts, not to repulsion (Fried & Boxer 2017)', MAIN,
     'paid for by distal binding', True),
    # Discussion places the exchange decay (2.79 and 3.80 per A, both already in Results) beside
    # Klinman & Kohen's ~23 per A for hydrogen tunneling; the 23 is THEIR quoted estimate.
    ('tunneling decay constant is attributed to Klinman & Kohen', MAIN,
     'Klinman and Kohen quote a decay constant of about $23$ per', True),
    ('the decay comparison is framed as mass, not model', MAIN,
     'The gap is one of mass rather than of model', True),
    ('density figure folded in', APP, 'fig6_pimd_density.pdf', True),
    ('convergence figure folded in', APP, 'figED2_bead_convergence.pdf', True),
    ('no Supplemental Material pointers left', MAIN, 'Supplemental Material Sec', False),
    ('cover letter carries the current title', COVER,
     'Exponential locality and coordinate\ndependence of Pauli exchange repulsion', True),
    ('cover letter drops the bound title', COVER,
     'An exponential bound on Pauli confinement', False),
    ('cover letter drops the framing-A title', COVER,
     'Fragment-separation curvature is not', False),
    ('no document keeps the old title', SUP, 'locality of direct exchange', False),
    # Title changed 2026-09-24: "bound" asserted an inequality that main.tex's own Limits
    # paragraph disclaims ("we do not claim the first bounds the second").
    ('main text carries the current title', MAIN,
     'Exponential locality and coordinate dependence of Pauli exchange repulsion', True),
    ('the title no longer claims a bound', MAIN,
     'An exponential bound on Pauli confinement', False),
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
    # 69, not 101: the five negative systems were sampled 16+16+16+16+5 times; 101 is all seven
    ('abstract carries the ensemble result with the right denominator', MAIN,
     'all $69$ configurations sampled from those five systems', True),
    ('body carries the ensemble result', MAIN, 'No system changes class', True),
    # K_perp^int was negative definite in six of seven ONLY at the static geometries.
    # Over the ensemble it is five. The qualifier is the whole point and must survive.
    ('total interaction six-of-seven is qualified as geometry-specific in the main text', MAIN,
     'six of the seven systems on these configurations', True),
    ('body qualifies the total-interaction count as geometry-specific', MAIN,
     'six of the seven systems on these configurations', True),
    # a mean of sorted eigenvalues is not the spectrum of a mean matrix: report frequencies
    ('body reports ensemble per-configuration frequencies, not a class count', MAIN,
     '$10$ of $16$ for V750A, with L754A positive in all $16$', True),
    ('appendix reports per-configuration frequencies, not a class for the mean matrix', APP,
     'rather than a class for the averaged matrix', True),
    ('appendix does not infer an indefinite mean matrix from a positive mean eigenvalue', APP,
     'so it is indefinite in the mean', False),
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
    # 28% is WT's SMALLER eigenvalue only. On the same substitution the larger eigenvalue
    # moves 130.4% and the spectral norm 43.6%, so it is not a universal error bar.
    ('the 28% figure is scoped to WT\'s smaller eigenvalue', MAIN,
     "it is the change in WT's \\emph{smaller} eigenvalue", True),
    ('no universal fragment error bar is attached to every magnitude', MAIN,
     'Every transverse magnitude quoted here should therefore be read', False),
    ('appendix states the sign is invariant across settings', APP,
     'Nothing tested reverses the classification', True),
    ('appendix names the donor fragment as the dominant sensitivity', APP,
     r'the donor fragment at $27.6\%$', True),
    ('appendix reports SAPT2+ preserving the sign', APP,
     'leaves its sign and the\nwhole argument intact', True),
    ('appendix records the wall truncation as not load-bearing', APP,
     'is not\nload-bearing', True),
    ('appendix keeps ALMO-EDA declared as unexecuted', APP,
     'ALMO-EDA on the native pair was not part of this sweep', True),
    ('appendix reports the gradient so the reader can check it', APP,
     'separation working rather than failing', True),

    # ---- framing B: the exponential bound leads the paper (2026-09-23) ----
    ('abstract poses range AND coordinate, not one distance', MAIN,
     'a question about range and about which displacement', True),
    ('abstract does not reduce the question to one distance', MAIN,
     'is a question about a single distance', False),
    # The abstract asserted an exclusion the Limits paragraph disclaims.
    ('abstract scopes the field estimate to the model', MAIN,
     'a one-electron hard-wall estimate of scale, not an exclusion', True),
    ('abstract does not assert the exclusion', MAIN,
     'Confinement is therefore not the\ncatalytic field', False),
    ('abstract keeps the measured-not-assumed discipline', MAIN,
     'At the measured contact distances,', True),
    ('exponential section precedes the coordinate section', MAIN,
     'How far direct exchange reaches', True),
    ('discussion separates the model estimate from the measured content', MAIN,
     'a modelling assumption rather than an inequality', True),
    ('discussion does not claim a generalised negative answer', MAIN,
     'negative for a reason that generalises beyond this enzyme', False),
    ('discussion no longer calls fragment sensitivity outstanding', MAIN,
     'whether they survive changes of fragment size and electronic treatment,\nis what is left', False),
    # Cover letter rewritten 2026-09-24. The old phrasing paired the biological-physics case
    # with "a bound rather than a correlation" and "We show it cannot be", which the manuscript
    # no longer claims. The case is now made on the measurement practice plus the honest null.
    ('cover letter argues the biological-physics case, not a methods case', COVER,
     'Any group\nstudying enzymatic hydrogen transfer can apply it to its own descriptor', True),
    ('cover letter gives the single-centre intuition', COVER,
     "costs $U'(r)/r<0$", True),
    ('cover letter does not claim an exclusion', COVER,
     'we do not claim to have excluded exchange-mediated contributions to catalysis', True),
    ('cover letter does not say the total matrix is uncomputed', COVER,
     'total\nnuclear restoring matrix is not computed', False),
    ('cover letter does not claim the conclusion generalises', COVER,
     'That conclusion generalises past lipoxygenase', False),
    ('cover letter reports the native donor, not the methane surrogate', COVER,
     'substrate fragment as the hydrogen donor rather than a small-molecule stand-in', True),

    # The letter must name the force-constant-decomposition precedent. A referee who knows
    # Duarte will otherwise read the novelty pitch as overclaimed against the manuscript,
    # which does cite it.
    ('cover letter names the force-constant decomposition precedent', COVER,
     'Duarte and Bruns did it for interacting quantum atoms in 2022', True),
    ('cover letter leads on the coordinate result', COVER,
     'one from displacing the hydrogen atom the cavity encloses, do not agree', True),

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
     'The apparent significance of the endpoint estimator is therefore an umbrella-window-spacing artefact', True),
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
     'solved that same single open paraboloidal boundary directly', True),

    # Two different matrices give two different counts and they must not be swapped.
    # K_perp^exch: smaller eigenvalue negative in 6 of 7 ensemble means (all but L754A), over
    # 16+5+16+16+16+16+16 = 101 configurations.  K_perp^int: negative DEFINITE in 6 of 7 at the
    # static geometries but only 5 of 7 over the ensemble, V750A's larger eigenvalue averaging
    # +0.0091 N/m.  Verified against stage5_ensemble_panel.json on 2026-09-23.
    ('the ensemble five-of-seven class count is gone', MAIN,
     'over the thermal ensemble it is five of seven', False),
    ('static negative-definite count carries the geometry qualifier', MAIN,
     'negative definite in six of the seven systems \\emph{at these geometries}', True),
    ('the anti-confining eigenvalue count is not left unqualified', APP,
     'anti-confining in six of seven systems with the methane donor and five of seven with the native one', True),

    # Verified against references/markland2018_arxiv.pdf on 2026-09-23.  The source says "a typical
    # system containing O-H covalent bonds", not X-H, and it does NOT state a
    # P ~ hbar*w_max/kT criterion anywhere; that inference is ours.  Both had been folded into
    # a single \\cite{markland2018}.  O-H is the higher frequency, so their P=32 overstates
    # what our C-H bond needs, which makes the self-criticism conservative rather than inflated.
    ('the P=32 requirement is attributed to O-H as the source states', MAIN,
     'simple structural properties of O--H bonds at room temperature requires', True),
    ('the P=32 requirement is not generalised to X-H', MAIN,
     'structural properties of X--H bonds', False),
    ('the bead-count criterion is marked as our inference, not the source\'s', APP,
     'is ours rather than theirs', True),
    ('the criterion is not presented as the accepted one', APP,
     'The accepted criterion is that $P$ must be a low multiple', False),

    # Verified against references/kohen2015.pdf on 2026-09-23.  The quotation is verbatim (only
    # sentence-initial "From" lowercased for embedding) and "orthogonal" occurs exactly ONCE in
    # the whole paper, in a concluding summary, so "asserted qualitatively without an associated
    # observable" is exact.  But the passage had said we make his stiffening picture quantitative,
    # when our intermolecular transverse curvature is anti-confining in six of seven: opposite
    # sign.  Claiming to quantify an antecedent we actually reverse both misdescribes it and
    # undersells the result.
    # Kohen's statement concerns the whole enzymatic system near the TS; ours is one wall
    # fragment's intermolecular contribution at restrained reactant-like geometries. Neither
    # confirming nor reversing him is the defensible reading.
    ('the Kohen antecedent is neither confirmed nor reversed', APP,
     'neither a confirmation nor a refutation of it', True),
    ('we do not claim to reverse Kohen', APP,
     'Making it quantitative reverses its sign', False),

    # ---- external audit responses, 2026-09-24. Each verified independently before acting.
    # A norm ratio is not the fraction along every direction; the generalised eigenproblem
    # K_exch v = lambda K_tot v gives extrema -0.097% to +0.283% (reproduced exactly).
    ('the exchange fraction reports directional extrema, not only the norm ratio', MAIN,
     'whose extrema over the tested transverse plane run from', True),
    # A constrained two-point energy difference is not an activation barrier nor a bound on one;
    # 3 of 10 optimizations converged and the reference-clamp values are extrapolations.
    ('the barrier comparison does not claim an upper bound', MAIN,
     'not automatically an activation barrier nor an upper bound on one', True),
    ('the barrier comparison does not claim to remove circularity', MAIN,
     'removes the circularity from the choice of', False),
    ('the reference-clamp barrier is labelled an extrapolation', MAIN,
     'by extrapolation beyond the computed range', True),
    # slo_xi0_full.json stores per-system/per-clamp MEANS; WT alone reaches 2.356 A.
    ('the cavity interval is labelled an interval of means', MAIN,
     'per-system means of $2.42$ to $2.95', True),
    # The stage-5 selector takes a strided prefix, not a spread across the trajectory.
    ('the stage-5 sampling prefix is disclosed', APP,
     'takes a strided\n\\emph{prefix} of each saved trajectory', True),
    ('stage-5 configurations are not called independent', APP,
     'on independent configurations', False),
    # HF treats polarization self-consistently; it lacks correlation dispersion.
    ('HF is not said to omit induction', APP,
     'SAPT0 contains induction and dispersion\nthat Hartree-Fock omits', False),
    # The primary-data Zenodo record does not exist yet; only the code record does.
    ('the primary data are not claimed to be already deposited', APP,
     'are a separate Zenodo record cross-linked to the code', False),

    # ---- 2026-09-24 external audit, second round. Each verified before acting.
    # Saved Mulliken data contradict the old diagnosis: substrate spin runs 0.002 -> 1.055 while
    # Fe falls 4.17 -> 3.77, in 32 of 75 outputs above 0.5. The scan DOES change character.
    ('the proton scan is not said to retain Fe(III) throughout', APP,
     'retaining Fe(III) character', False),
    ('the proton-scan spin trace is reported', APP,
     'Mulliken spin on iron falls from', True),
    # ORCA records: 7 unconverged jobs fail energy AND max-gradient, and only 2 of 10 meet
    # every criterion (rda290_react is accepted under the alternative gradient rule).
    ('the convergence description matches the ORCA records', APP,
     'only two of\nthose meet every individual criterion', True),
    ('the barrier is not called an upper bound', APP,
     'each barrier is an upper bound on a fully relaxed path', False),
    ('the reference-clamp figures are labelled extrapolation', APP,
     'by \\emph{extrapolation} beyond the', True),
    # methane panel: I552A is 2.65 fit SE from zero, so "16 to 48" was wrong
    ('the methane sigma range is not overstated', MAIN, '16$ to $48', False),
    # f is a participation measure; small f does not locate the flux
    ('f is not said to prove rare configurations carry the rate', APP,
     'rate is carried by a rare high-potency', False),
    # averaging sorted eigenvalues is not the spectrum of the averaged matrix
    ('no system is classified by its mean matrix', MAIN, 'indefinite in the mean', False),
    ('the averaging caveat is stated', APP,
     'coincide with the spectrum of the averaged Hessian only when the eigenvectors do not rotate', True),
    # the transverse ZPE is one channel, not the only one
    ('the ZPE channel is not claimed to be the only route', MAIN,
     'effect only by changing the curvature of the transverse potential', False),
    # nonorthogonal diabatic states need an overlap-corrected coupling
    ('the overlap-corrected coupling is given', APP, 'L\\"owdin orthogonalization', True),
    # the m-dependent wavefunction needs its regularity factor
    ('the parabolic wavefunction carries its regularity factor', APP,
     '(\\rho_1\\rho_2)^{|m|/2}', True),
    # descriptors were tested; the hypothesis was not
    ('the hypothesis is not described as tested and unsupported', MAIN,
     'confinement hypothesis are therefore tested and', False),
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

    # Duarte force-constant decomposition: the novelty scoping and the numbers derived
    # from scripts/duarte_projection_check.py. The 2022 paper decomposes an angular
    # bending force constant, so a bare claim to the transverse coordinate is stale.
    ('the transverse coordinate is not claimed as novel', MAIN,
     'neither the second derivative nor a coordinate transverse to a bond is ours to claim', True),
    ('the IQA bend counterexample is stated', MAIN,
     'it is the intra-atomic term that is negative while the exchange contribution is '
     'negligible or slightly positive', True),
    ('the generalized extrema are attributed to their method', MAIN,
     'against the Hartree-Fock total and $-0.117\\%$ to $+0.352\\%$ against the B3LYP one', True),
    ('the additive shares are the HF span', MAIN,
     'they run from $-0.067\\%$ to $+0.279\\%$ at Hartree-Fock', True),
    ('the additive comparison is four versus five, not four versus six', APP,
     'which are resolved negative in five systems', True),
    ('the appendix does not claim V750A lost a negative native component', APP,
     'removes the negative share in I552A and V750A', False),
    ('the additive share span is quoted for both totals', APP,
     '$-0.067$ to $+0.279\\%$ against the Hartree-Fock total and $-0.084$ to $+0.348\\%$ '
     'against B3LYP', True),
    # The script pointer was removed from the article at the author's request, so the
    # check that required it now forbids it instead: a removed string must not creep back.
    ('the article does not name the projection script', APP,
     'duarte\\_projection\\_check', False),
    # Inline repo paths were removed throughout: a reader cannot act on a bare working
    # filename, the Code Availability statement already points at the whole repository,
    # and two of the files named were not even tracked in it.  Software names, ORCA
    # keywords and library calls are not paths and are unaffected.
    ('no script filename is printed in the article', APP, '.py}', False),
    ('no results path is printed in the article', APP, 'texttt{results/', False),
    ('the unfindable demo scan is not cited', APP, 'sapt\\_demo', False),
    ('the unpublished revision-audit script is not cited', APP,
     'check\\_coordinate\\_claims', False),

    # ---- revision-2 audit, 2026-09-24 afternoon.  Each was a statement the manuscript's own
    # revised passages withdrew while a duplicate went on asserting it.  Guard the ABSENCE of
    # the withdrawn form, not only the presence of the correction: a positive-string check
    # cannot detect the opposite claim surviving somewhere else in the document.
    ('the withdrawn Fe(III)-retention claim is gone', APP,
     'retained Fe(III) character across the whole scan', False),
    ('the scan is not called single-diabatic', APP, 'single-diabatic-surface', False),
    ('the scan is described as unconstrained adiabatic', APP,
     'unconstrained single-adiabatic-surface construction', True),
    ('the stale three-of-ten convergence count is gone', MAIN,
     '$3$ of the $10$ optimizations met every', False),
    ('the mean/median orderings are not claimed to agree', APP,
     'and the two orderings agree', False),
    ('the secondary association is placed at the reference condition', MAIN,
     'appears at the reference condition rather than the compressed one', True),
    ('the non-reactive-location wording is gone', MAIN,
     'the inverted sign and the non-reactive location', False),
    ('thermal motion is not said to increase anti-confinement', APP,
     'anti-confinement away; it increases it', False),
    ('the constrained pilot is not called an electronic barrier', MAIN,
     'the electronic barrier falls by', False),
    ('the constrained pilot is named as an endpoint energy difference', MAIN,
     'constrained endpoint energy difference', True),
    ('one constrained path is not read as excluding transfer', APP,
     'so no transfer occurs at the equilibrium separation at all', False),
    ('the biased snapshot mean is not called an equilibrium mean', APP,
     'at the equilibrium umbrella-sampling', False),
    ('f is not called the share of the ensemble carrying flux', APP,
     'share of the equilibrium ensemble that carries isotope-$i$ flux', False),
    ('the FC inversion does not assert a small f', APP,
     'this places the wild-type $f\\!\\ll\\!1$', False),
    ('the ZPE channel is not called a maximum isotope effect', APP,
     'the largest isotope effect one variant can gain', False),
    ('mutagenesis is not said to report the separation curvature', MAIN,
     'The exchange-repulsion curvature a cavity-lining mutation reports', False),
    ('the descriptor is presented as our own computational proposal', MAIN,
     'We tested a fragment-separation curvature as a computational descriptor', True),
    ('the hydrogen is not called mid-transfer', MAIN, 'is mid-transfer in an', False),
    ('the exponential form is not called established in the active site', MAIN,
     'exponential form is established in a real active site', False),
    ('the abstract scopes the exponential rate to the solved geometries', MAIN,
     'at a rate fixed in both solved geometries', True),
    ('the Duarte comparison is not said to remove the partition explanation', MAIN,
     'the agreement removes one explanation we would otherwise owe a reader', False),
    ('the sphere kT range is not the superseded 1.4 to 8.2', APP,
     'The spherical geometry gives $1.4$ to $8.2$', False),
    ('the restraint offsets are the raw-series values', APP, '$+0.2289\\,\\text{\\AA}$', True),
    ('the superseded restraint offset is gone', APP, '$+0.223\\,\\text{\\AA}$', False),
    ('the permutation test is not called exact', MAIN, 'exact permutation', False),
    ('the stage-5 grid retention limit is stated', APP,
     'cannot be independently refitted from it', True),
    ('the cover letter does not generalise the 28 per cent', COVER,
     'setting a $28\\%$ uncertainty on the magnitude', False),
    ('the cover letter reports both totals', COVER,
     'against the Hartree-Fock total and $-0.117\\%$ to $+0.352\\%$', True),
    ('the cover letter does not claim the complete record is public', COVER,
     'The complete input, output and analysis record for every number', False),
    ('the version DOI is cited', MAIN, '10.5281/zenodo.22947653', True),

    # ---- equation audit, 2026-10-29.  Every display equation must fit one column.
    ('no widetext in the main text', MAIN, 'begin{widetext}', False),
    ('no widetext in the appendices', APP, 'begin{widetext}', False),
    # eq:HSrate carried unindexed level energies while eq:kchannels and eq:SEqH index them
    ('HSrate indexes the vibronic levels by isotope', APP,
     'varepsilon_{\\nu}^{(i)}-\\varepsilon_{\\mu}^{(i)}\\big)^{2}}{4\\lambda(\\mathbf R)k_{B}T}\\right].', True),
    # eq:Efield mixed r and R for the same distance
    ('Efield uses one symbol for the distance', APP, '(r\\times10^{-10})', False),
    # the dipole asymptote's large-xi0 behaviour is stated with its 50-digit numbers
    ('the asymptote numbers at large xi0 are stated', APP,
     'reaches $1.019$ at $25$ and is still $1.015$ at $50$', True),
    ('the asymptote error direction is stated the right way round', APP,
     'past the crossing the leading form underestimates the exact value', True),
    ('the reversed direction wording is gone', APP,
     'the leading form overshoots rather than undershoots', False),

    # ---- 2026-10-29 audit.  Absence guards for every statement replaced in that pass.
    ('the abstract counts 69 configurations for the five negative systems', MAIN,
     'all $69$ configurations sampled from those five systems', True),
    ('the abstract does not claim 101 configurations for five systems', MAIN,
     'every one of $101$ sampled configurations', False),
    ('the wall check is not extended to I553A', MAIN,
     'grid and wall checks extended to L754A and I553A', False),
    ('the appendix does not run all six settings on L754A', APP,
     'Across all six settings $\\lambda_1$ stays', False),
    ('Figure 5 caption uses the plotted geometry distance', MAIN,
     '$3.46\\,\\si{\\angstrom}$ away in this geometry', True),
    ('Figure 5 caption does not carry the equilibrium-campaign 4.8', MAIN,
     'polar and\n$4.8\\,\\si{\\angstrom}$ away', False),
    ('the ensemble is reported by per-configuration frequency, not a class count', MAIN,
     'it is five of seven, V750A being the', False),
    ('the methane Methods text does not call the wall scan transverse', APP,
     'obtained\nfrom displacements \\emph{transverse} to that axis', False),
    ('the methane scan is named a fragment-separation curvature', APP,
     'the fragment-separation curvature reported as $k_{\\mathrm{exch}}^{\\mathrm{bio}}=2a_{2}$', True),
    ('the proton scan is not called QM/MM', MAIN, 'QM/MM proton', False),
    ('the proton scan is not called QM/MM in the appendix', APP, 'QM/MM proton', False),
    ('the overlap term is not said to have to carry the ladder', MAIN, 'must carry the ladder', False),
    ('the longitudinal comparison is not a failed test', MAIN, 'the wall contact fails it', False),
    ('no pocket radius is inferred from the transverse Hessian', MAIN,
     'no pocket radius is inferred from it', True),
    ('the native pair is not called open-shell', MAIN, 'open-shell native pair', False),
    ('the range-weight gap is 0.39 A', APP, 'about $0.39\\,\\si{\\angstrom}$ rather than $0.75$', True),
    ('the range-weight factor at the measured gap is 4.3', APP, 'the measured $0.39\\,\\si{\\angstrom}$ into $4.3$', True),
    ('the e-folding length is a0/2', APP, 'e-folding length is $a_0/2=0.26', True),
    ('the half-angstrom decay length is gone', APP, 'decay length of half an angstrom', False),
    ('the cavity is not said to be unable to recover the factor', APP, 'by weight of numbers, and it cannot', False),
    ('the paraboloid prefactor argument keeps u0', APP, '\\epsilon\\simeq u_0e^{-u_0}\\simeq2d', True),
    ('Table VIII I538A native value is 6.53', APP, '$\\phantom{1}6.53$ & $100$', True),
    ('Table XIV L754A residual is 8.4e-6', APP, '$0.1633$ & $8.4\\times10^{-6}$', True),
    ('Table XVI text does not call the differences barriers', APP, 'Barriers are stable', False),
    ('LaCour is cited as the published article', Path(str(COVER).replace('cover-letter.tex', 'pauli-references.bib')),
     'J. Chem. Theory Comput.",\n  volume  = "22"', True),
    ('the Duarte range names the ammonia donors near 2 per cent', MAIN, 'with the ammonia donors near $2\\%$', True),
    ('the cover letter uses the 69 denominator', COVER, '$69$ in all for the five negative ones', True),
    ('the cover letter does not claim 101 for the resolved systems', COVER,
     'persists in every one of $101$ configurations', False),

    # A draft-era convention announced that provisional results were flagged
    # [pending completion of the validation study].  The macro was defined once, used once,
    # and marked NO result: every run had finished.  It told a referee the paper contained
    # provisional numbers that do not exist.
    ('no provisional-results convention is announced', APP,
     'Results that depend on the running simulations are marked', False),
    ('the pending macro is not defined', MAIN, 'newcommand{\\pending}', False),
    # the protocol headings carried a hard-coded "6.N" from a structure this appendix no
    # longer has, printed inside a lettered appendix
    ('the protocol headings are not hard-numbered', APP, 'subsection*{6.', False),
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
    skipped = 0
    for desc, f, text, want in CHECKS:
        if f not in cache:
            # A source this repository does not ship (the cover letter is author-editor
            # correspondence and names suggested referees, so it is not published).
            # Skip its checks rather than abort: everything else is still checkable.
            cache[f] = flat(f.read_text()) if f.exists() else None
        if cache[f] is None:
            skipped += 1
            print(f'  [SKIP] {desc:<{width}}  ({f.name}: not in this checkout)')
            continue
        present = flat(text) in cache[f]
        ok = (present == want)
        bad += not ok
        print(f'  [{"OK " if ok else "FAIL"}] {desc:<{width}}  ({f.name}: '
              f'{"present" if present else "absent"})')
    ran = len(CHECKS) - skipped
    if ran == 0:
        # Do not print "0/0 passed": nothing was verified, and a reader skimming the
        # last line must not mistake that for success.
        print(f'\n  NOTHING CHECKED: all {skipped} checks need manuscript sources that this '
              f'checkout does not contain.\n  They are not published; see README. Point '
              f'PAULI_ROOT at a tree that has prxlife/ to run them.')
        return bad
    tail = f'  ({skipped} skipped: source not in this checkout)' if skipped else ''
    print(f'\n  {ran-bad}/{ran} consistency checks passed{tail}')
    return bad


if __name__ == '__main__':
    sys.exit(main())

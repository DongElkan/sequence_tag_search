""" Analysis of synthetic peptides """
import os
import sys
import re
import collections
import itertools
import json
import typing
import numpy as np
from operator import attrgetter
from _bisect import bisect_left

from pepfrag import constants, Peptide, ModSite
from rPTMDetermine.peptide_spectrum_match import PSM
from rPTMDetermine.readers import PTMDB

from feature_extraction import FeatureExtraction

ptmdb = PTMDB()
featuregetter = FeatureExtraction()

Feature_names = (
    "NumPeaks", "TotInt", "PepMass", "Charge", "FracIon", "FracIonInt",
    "NumSeriesbm", "NumSeriesym", "NumIona", "NumIonynl", "NumIonbnl",
    "FracIonIntb_c1", "FracIonIntb_c2", "FracIonInty_c1", "FracIonInty_c2",
    "FracIon20pc", "NumIonb", "NumIony", "FracIonInty", "FracIonIntb",
    "MatchScore", "SeqTagm"
)

target_pairs = {
    "Kmod_Biotinyl": "Kmod_Biotin", "Kmod_Propionyl": "Kmod_Propion",
    "Kmod_Ubiquitinyl": "Kmod_Glygly"
}

SynMatch = collections.namedtuple(
    "SynMatch", ["seq", "mods", "charge", "num_ions", "prec_mz",
                 "num_seqtags", "max_tag", "ion_index", "delta_mass",
                 "nterm_left", "cterm_left"],
    defaults=[None] * 11
)

SeqCorrect = collections.namedtuple(
    "SeqCorrect", ["rt_idx", "ist_idx", "ist_seq", "mods"],
    defaults=[None] * 4
)

ProcMatch = collections.namedtuple(
    "ProcMatch", ["seq_prefix", "mass_prefix", "mods_prefix", "seq_proc",
                  "mods_proc", "seq_suffix", "mass_suffix", "mods_suffix"],
    defaults=[None] * 8
)

TagPrecursor = collections.namedtuple(
    "TagPrecursor",
    ["mz_tag", "pmass", "length", "seq_term", "ion_type", "mods_tag",
     "pseq", "index", "tag", "pmods", "index_tag"],
    defaults=[None] * 11
)


def _common_substr(seq1: str, seq2: str) -> typing.List[str]:
    """ Longest common substring between two sequences. """
    prec_seq = np.array(list(seq2))
    # number of characters
    n = prec_seq.size
    # initializations, only use last and current arrays for updating
    # they are nested lists with inner list of tuples (i, j):
    # index of common residue in (seq1, seq2)
    S: typing.List[typing.List[tuple]] = [[] for i in range(n)]
    S0: typing.List[typing.List[tuple]] = [[] for i in range(n)]
    # the first element in first sequence
    ix, = np.where(prec_seq == seq1[0])
    for i in ix:
        S0[i].append((0, i))

    # search common substrings
    str_pre, str_next, common_str_index = ix.tolist(), [], []
    for i in range(1, len(seq1)):
        ix, = np.where(prec_seq == seq1[i])
        if ix.size > 0:
            if ix[0] == 0:
                S[0].append((i, 0))
                ix = ix[1:]
            for j in ix:
                S[j] = S0[j - 1] + [(i, j)]
                if len(S[j]) > 1:
                    str_next.append(j)
        # store common substrings
        if str_pre:
            common_str_index += [
                S0[j] for j in set(str_pre) - set(j - 1 for j in str_next)
                if S0[j] not in common_str_index and S0[j]
            ]
        # backup current match as previous match for the next and clear
        # current match
        S0, str_pre = S.copy(), str_next.copy()
        S, str_next = [[] for i in range(n)], []
    # the last element
    if str_pre:
        common_str_index += [S0[j] for j in str_next
                             if S0[j] not in common_str_index and S0[j]]

    return common_str_index


class _Spectrum():
    """ Composition from psm object """
    def __init__(self, spectrum: np.ndarray,
                 precursor_mz: float = None) -> None:
        self.spectrum = spectrum
        self.precursor_mz = precursor_mz

    def denoise(self,
                num_retain: int = 6,
                noise: float = 0.005) -> np.ndarray:
        """
        Denoise mass spectrum based on the rule 6 top peaks in
        100 Da window.
        """
        peaks = self.spectrum
        # remove very low intensity peaks
        peaks = peaks[peaks[:, 1] / peaks[:, 1].max() >= noise]
        deisotope_tol = 0.2
        n_windows = int((peaks[-1][0] - peaks[0][0]) / 100.) + 1

        i0, npeaks = 0, peaks.shape[0]
        denoised_peaks = []
        # denoise spectrum
        for win in range(n_windows):
            # upper limit mz in current window
            max_mz = peaks[0][0] + (win + 1) * 100
            for i in range(i0, npeaks):
                if peaks[i][0] >= max_mz:
                    break
            if i == i0:
                continue

            # reset the starting peak index for next window.
            if i == npeaks - 1 and peaks[i][0] < max_mz:
                i = npeaks

            # deisotoping
            sub_peaks = self.deisotope(peaks[i0:i], tol=deisotope_tol)
            # sort peaks based on intensities
            sub_peaks = sub_peaks[sub_peaks[:, 1].argsort()[::-1]]
            denoised_peaks += sub_peaks[:num_retain].tolist()

            # reset the starting index
            i0 = i

        denoised_peaks = np.array(denoised_peaks)
        self.spectrum = denoised_peaks[denoised_peaks[:, 0].argsort()]

        return self.spectrum

    def deisotope(self, peaks: np.ndarray, tol: float = 0.2) -> np.ndarray:
        """
        Deisotope peaks.

        Args
        ----
        peaks: np.ndarray
            Mass spectral peaks for deisotoping.
        tol: float
            Tolerance in dalton to detect isotope peaks.

        Return
        ------
        Deisotoped peaks

        """
        if peaks.shape[0] == 1:
            return peaks

        # deisotoping
        deiso_peaks, rm_index = [], set()
        for i, peak in enumerate(peaks):
            if i in rm_index:
                continue

            # iterate through peaks to detect isotopes by assuming
            # different charge states
            iso_index = []
            for c in [1, 2]:
                peak0, j0, iso_index_c = peak, i, [i]
                while True:
                    has_isotope = False
                    for j, peak1 in enumerate(peaks[j0:]):
                        # TODO: use similarity comparison to detect
                        # isotopic distribution.
                        if (abs((peak1[0] - peak0[0]) * c - 1) <= tol
                                and 1 - peak1[1] / peak0[1] >= 0.2):
                            iso_index_c.append(j0 + j)
                            has_isotope = True
                            break
                    if not has_isotope:
                        break
                    j0 += j + 1
                    peak0 = peak1
                # maximal length as the detected isotopes
                if len(iso_index_c) > len(iso_index):
                    iso_index = iso_index_c

            # remove isotopic peaks
            rm_index.update(iso_index)
            deiso_peaks.append(peaks[iso_index[0]])

        return np.array(deiso_peaks)

    def extract_sequence_tags(self, tol: float = 0.2) -> None:
        """ Search tags corresponding to synthetic peptides. """
        # pairwise m/z differences
        diff_mz = self.spectrum[:, 0] - self.spectrum[:, 0][:, np.newaxis]
        # amino acid mass
        aas = "".join(sorted(constants.AA_MASSES.keys()))
        aam = np.array([constants.AA_MASSES[a].mono for a in aas])
        aam = aam[:, np.newaxis]
        aa_min, aa_max = aam.min(), aam.max()
        # get sequence tags from the spectrum
        seq_tags, index_tags, assigned = [], [], set()
        for i in range(self.spectrum.shape[0]):
            if i in assigned:
                continue

            # search tags
            alive_ix, alive_seqs = [[i]], [[]]
            while True:
                curr_ix, curr_seq = [], []
                for kx, aax in zip(alive_ix, alive_seqs):
                    # get next residue
                    k = kx[-1]
                    df_k = diff_mz[k][k+1:]
                    jx, = np.where(
                        (df_k <= aa_max + tol) & (df_k >= aa_min - tol)
                    )

                    # filter the m/z differences by min and max AA mass
                    ix, tags = [], []
                    if jx.size > 0:
                        aa_diff = np.absolute(df_k[jx] - aam)
                        for aa, diff in zip(aas, aa_diff):
                            _ix = jx[diff <= tol]
                            if _ix.size > 0:
                                ix += _ix.tolist()
                                tags += [aa] * _ix.size
                    if ix:
                        curr_ix += [kx + [j + k + 1] for j in ix]
                        curr_seq += [aax + [a] for a in tags]
                    elif len(aax) > 0:
                        seq_tags.append("".join(aax))
                        index_tags.append(kx)

                # if no new residue is found, stop the search
                if not curr_ix:
                    break
                alive_ix, alive_seqs = curr_ix, curr_seq

            # assigned indices
            assigned.update(itertools.chain(*index_tags))

        self.sequence_tags = seq_tags
        self.peak_index = index_tags

    def match_sequence(self, seq: str, mods: ModSite = None) -> np.ndarray:
        """ Match sequence. """
        # parse modifications to separate them into three parts
        self._parse_modification(mods)
        # precursor mass
        pmass = (sum(constants.AA_MASSES[a].mono for a in seq)
                 + constants.FIXED_MASSES["H2O"])
        if mods:
            pmass += sum(_m.mass for _m in mods)

        # match seq by common subsequences
        prec_tag, assigned = [], set()
        # matches as b ions
        index_b = [(i, "b") for i, tag in enumerate(self.sequence_tags)
                   if any(tag[i:i+3] in seq for i in range(len(tag) - 2))]
        # matches as y ions
        seq2 = seq[::-1]
        index_y = [(i, "y") for i, tag in enumerate(self.sequence_tags)
                   if any(tag[i:i+3] in seq2 for i in range(len(tag) - 2))]
        # construct matches
        for i, yb_type in index_b + index_y:
            prec_tag_c = self._get_tag_precursor(
                seq, self.sequence_tags[i], ion_type=yb_type
            )
            # unique matches
            pix = self.peak_index[i]
            for t in prec_tag_c:
                mz = self.spectrum[pix[t.index_tag]][0]
                if (mz, t.seq_term, t.ion_type, t.index, t.length) in assigned:
                    continue
                prec_tag.append(t._replace(mz_tag=mz, pmass=pmass, pmods=mods))
                assigned.add((mz, t.seq_term, t.ion_type, t.index, t.length))

        return prec_tag

    def _get_tag_precursor(self, seq, tag, ion_type=None):
        """
        Get fragment precursor containing tags.

        Parameters
        ----------
        seq: str
            Parent sequence.
        ion_type: str
            Type of fragment ion.
        pmass: float
            Precursor mass.

        Returns
        -------
        List of objects

        """
        seq_m = seq if ion_type == "b" else seq[::-1]
        ns = len(seq)
        mod_nterm, mod_cterm, mods_ints = self._parsed_mods
        # common subsequences
        precursors = []
        subseqs = _common_substr(tag, seq_m)
        for sub_index in subseqs:
            n = len(sub_index)
            # the number of tags should be larger than 2.
            if n <= 2:
                continue
            # index of common subseq starting and ending
            (_, j0), (i1, j1) = sub_index[0], sub_index[-1]
            # fragment precursors, subsequence and modifications
            term = seq_m[:j0+n]
            if ion_type == "y":
                term, j1 = term[::-1], ns - j1 - 1
                mods = (mod_cterm + [m._replace(site=int(m.site-j1))
                                     for m in mods_ints if m.site > j1])
            else:
                mods = mod_nterm + [_m for _m in mods_ints if _m.site <= j0+1]
            # save the information
            seq_frag = {
                "seq_term": term, "length": n, "tag": sub_index,
                "mods_tag": mods, "pseq": seq, "ion_type": ion_type,
                "index_tag": i1 + 1, "index": j1
            }
            precursors.append(TagPrecursor(**seq_frag))

        return precursors

    def _parse_modification(self, mods: ModSite = None):
        """ Parse modifications. """
        # parse modifications
        mod_nterm, mod_cterm, mods_ints = [], [], []
        if mods is not None:
            for _mod in mods:
                if isinstance(_mod.site, int):
                    mods_ints.append(_mod)
                elif _mod.site == "nterm":
                    mod_nterm.append(_mod)
                else:
                    mod_cterm.append(_mod)
        self._parsed_mods = (mod_nterm, mod_cterm, mods_ints)


class CorrectSynthesisMatch():
    """ Correct false positive matches using synthetic peptides. """
    def __init__(self, validator, tol=0.1):
        # load targets
        self._get_targets()
        # load synthetic peptides
        self._get_synthetic_peptides()
        # load artifacts
        self._get_artifact()
        # validator
        self.validator = validator
        self.tol = tol

    def correct_psms(self, psm):
        """ Identify the mass spectrum from peptide synthesis error. """
        # match to synthetic peptides
        # synthetic peptide targets
        syn_peps = self.syn_peps[self._parse_raw_id(psm.data_id)]

        # annotate by synthetic peptides to get isobarics
        candidates = self._annotate_by_synthetic_peptides(psm, syn_peps)

        # matches to synthetic peptides
        matches = self._match_syn_peptide(psm, syn_peps)
        # refine the matches
        if matches:
            for match in matches:
                candidates += self._correct_psm(match)
            if candidates:
                return self._validate_candidates(psm, candidates)

        # sequence tag search
        tag_matches = self._sequence_tag_search(psm, syn_peps)
        if tag_matches is None:
            return None
        # correct the mass of tag precursors
        pmz_eu = psm.spectrum.prec_mz - constants.FIXED_MASSES["H"]
        for match, tag in tag_matches:
            candidates_tag = self._correct_psm(match)
            # correct the mass of the rest of the peptide
            for pep in candidates_tag:
                m2 = self._tag_match_correct(pep, tag)
                dm = [(m2.delta_mass - pmz_eu * c, c) for c in range(2, 5)]
                candidates += self._correct_psm(m2._replace(delta_mass=dm))

        if candidates:
            return self._validate_candidates(psm, candidates)

        return None

    def _annotate_by_synthetic_peptides(self, psm, syn_peptides):
        """ Annotate the spectrum by corresponding synthetic peptide """
        mz = psm.spectrum.prec_mz
        pep_candidates = []
        for _seq, _mods in syn_peptides:
            for c in range(2, 5):
                pep = Peptide(_seq, c, _mods)
                _mz = pep.mass / c + constants.FIXED_MASSES["H"]
                if abs(_mz - mz) <= self.tol:
                    pep_candidates.append(pep)
        return pep_candidates

    def _match_syn_peptide(self, psm, syn_peptides, match=SynMatch):
        """ Match synthetic peptide forcely. """
        pmz_neu = psm.spectrum.prec_mz - constants.FIXED_MASSES["H"]
        # assign mass spectra using synthetic peptides
        matches = []
        for seq, mods in syn_peptides:
            # peptide project
            pep = Peptide(seq, 2, mods)
            mch = PSM(psm.data_id, psm.spec_id, pep, psm.spectrum)
            annotates, _ = mch.denoise_spectrum()

            # get annotations
            n, seq_ion_index = 0, collections.defaultdict(list)
            for ion, (_, ion_index) in annotates.items():
                if ion[0] in "yb" and "-" not in ion and ion.endswith("[+]"):
                    n += 1
                    seq_ion_index[ion[0]].append(ion_index)
            if n == 0:
                continue

            # maximum length of seq tags and the maximum ion index
            n_tag, max_ion_index = [], {"y": 0, "b": 0}
            for ion in seq_ion_index.keys():
                nion = len(seq_ion_index[ion])
                if nion > 1:
                    index_diff, i0 = np.diff(sorted(seq_ion_index[ion])), -1
                    tx, = np.where(index_diff > 1)
                    for i in tx:
                        n_tag.append(i - i0)
                        if i - i0 > 1:
                            max_ion_index[ion] = seq_ion_index[ion][i + 1]
                        i0 = i
                    # the end of the array
                    if nion - i0 > 2:
                        n_tag.append(nion - i0 - 1)
                        max_ion_index[ion] = max(seq_ion_index[ion])
                else:
                    n_tag.append(len(seq_ion_index[ion]))
                    if len(seq_ion_index[ion]) == 1:
                        max_ion_index[ion] = seq_ion_index[ion][0]

            # if number of ions in a sequence tag is gt 3 or more than
            # two sequence tags having number of ions equaling to 3
            if max(n_tag) >= 4 or n_tag.count(3) >= 2:
                # delta mass
                dms = [(pep.mass - pmz_neu * c, c) for c in range(2, 5)]
                # whether Terminus is tended to be modified
                nterm = False if max_ion_index["b"] > 0 else True
                cterm = False if max_ion_index["y"] > 0 else True
                matches.append(
                    match(seq=seq, mods=mods, charge=psm.charge, num_ions=n,
                          num_seqtags=n_tag, max_tag=max(n_tag),
                          ion_index=max_ion_index, delta_mass=dms,
                          nterm_left=nterm, cterm_left=cterm)
                )

        return matches

    def _correct_psm(self, match):
        """ Correct PSM. """
        # sparated match to consider subsequence only
        sepm = self._parse_matches(match)
        # get combinations and artifacts
        combs, combs_add, artifs, rp = self._restrict_refs(
            sepm.seq_proc, sepm.mods_proc,
            nterm=match.nterm_left, cterm=match.cterm_left
        )
        s = "".join(rp)
        # get the combinations of error loss
        candidates, unique_peps = [], set()
        for dm, c in match.delta_mass:
            if abs(dm) <= self.tol:
                candidates.append(Peptide(match.seq, c, match.mods))
                continue

            # all possible corrections
            corrects = self._correct_mass(s, dm, combs, combs_add, artifs)
            for corr in corrects:
                mods_c = []
                if corr.ist_idx is None:
                    seq_c = "".join(rp[corr.rt_idx]).upper()
                    # reset modifications
                    for mod in sepm.mods_proc:
                        if mod.site-1 in corr.rt_idx:
                            j = int(corr.rt_idx.index(mod.site-1))
                            mods_c.append(mod._replace(site=j+1))
                    # set up modification sites from corrections.
                    if corr.mods is not None:
                        mods_c += corr.mods
                else:
                    iseq = np.array(corr.ist_seq, dtype=str)
                    seq_c = "".join(
                        np.insert(rp[corr.rt_idx], corr.ist_idx, iseq)
                    )
                    # reset modifications
                    for mod in sepm.mods_proc:
                        if mod.site-1 in corr.rt_idx:
                            j = corr.rt_idx.index(mod.site-1) + 1
                            # if residue insert in current subsequence,
                            # reset modification site again.
                            j2 = sum(i < j for i in corr.ist_idx)
                            mods_c.append(mod._replace(site=int(j+j2)))

                seq, mods = self._reconstruct_peptide(sepm, seq_c, mods_c)
                # if the peptide has been identified, ignore it.
                pep = self._combine_mods(seq, mods)
                if pep not in unique_peps:
                    candidates.append(Peptide(seq, c, mods))
                    unique_peps.add(pep)

        return candidates

    def _correct_mass(self, seq, dm, res_combs, res_combs_add, artifacts):
        """
        Match artifact modifications and synthesis error.

        Parameters
        ----------
        seq: str
            Sequence for correction.
        dm: float
            Delta mass.
        res_combs: dict
            Combinations of residues.
        res_combs_add: dict
            Combinations of residues as additional residues
            for the correction.
        artifacts: dict
            Artifacts.

        """
        corrects = []
        # additionals
        add_res, res_mass = res_combs_add["residues"], res_combs_add["mass"]
        add_mod, mod_mass = artifacts["mods"], artifacts["mass"]
        # match residue combinations
        for i in range(max(res_combs.keys()) + 1):
            # i == 0 indicates no residue removed
            res_i = res_combs[i]["residues"] if i > 0 else [""]
            mass_i = res_combs[i]["mass"] if i > 0 else [0.]

            if dm > 0 and i > 0:
                # remove residues from subseq in candidates
                for res in res_i[np.absolute(mass_i - dm) <= self.tol]:
                    rt_ix, _ = self._remove_mass(seq, res)
                    if rt_ix is not None:
                        corrects += [SeqCorrect(rt_idx=ix) for ix in rt_ix]

            # further combination of residues and modifications
            for r1, m1 in zip(res_i, mass_i):
                # compensates from other residue combinations
                ix, = np.where(np.absolute(m1 - res_mass - dm) <= self.tol)
                compx = [(add_res[i], res_mass[i], "seq") for i in ix
                         if not set(r1) & set(add_res[i])]
                # compensates from modifications
                ix, = np.where(np.absolute(m1 - mod_mass - dm) <= self.tol)
                compx += [(add_mod[i], mod_mass[i], "mod") for i in ix]

                if compx:
                    # remove residues
                    ix, seqs = self._remove_mass(seq, r1)
                    if ix is not None:
                        # add up new residues or modifications
                        for res, m2, _type in compx:
                            corrects += self._add_mass(
                                seqs, ix, res, seq_type=_type, mass=m2
                            )

        return corrects

    def _remove_mass(self, seq, rm_seq):
        """ Remove residues rseq from seq. """
        if not rm_seq:
            return [list(range(len(seq)))], [seq]

        if len(seq) < len(rm_seq) or len(set(rm_seq) - set(seq)) > 0:
            return None, None

        # count the residues
        rseq_counts = collections.Counter(rm_seq)
        seq_counts = collections.Counter(seq)
        # if some of the residues not fulfill the counts in seq
        if any(seq_counts[r] < rseq_counts[r] for r in rseq_counts):
            return None, None

        # index of removed residues
        arr_seq, del_ix = np.array(list(seq), dtype=str), []
        for r in rseq_counts:
            ix, = np.where(arr_seq == r)
            del_ix.append(list(itertools.combinations(ix, rseq_counts[r])))

        # all combinations of residues to remove
        rt_index, rt_seqs, n = [], [], len(seq)
        for _comb in itertools.product(*del_ix):
            ix = set(itertools.chain(*_comb))
            rt_ix = [i for i in range(n) if i not in ix]
            rt_index.append(rt_ix)
            rt_seqs.append("".join(arr_seq[rt_ix]))

        return rt_index, rt_seqs

    def _add_mass(self, seqs, rt_index, add_seq, seq_type="seq", mass=None):
        """ Add residues to seq. """
        adds = []
        # simply insert add_seq to seq
        if seq_type == "seq":
            n = len(add_seq)
            for rt in rt_index:
                ix = range(len(rt) + 1)
                adds += [
                    SeqCorrect(rt_idx=rt, ist_idx=x, ist_seq=seq_sub)
                    for x in itertools.combinations_with_replacement(ix, n)
                    for seq_sub in itertools.combinations(add_seq, n)
                ]

        # add modifications to seq
        elif seq_type == "mod":
            _m, _t = add_seq[0], add_seq[1]
            # sites
            if _t not in constants.AA_MASSES:
                return [SeqCorrect(rt_idx=_ix, mods=[ModSite(mass, _t, _m)])
                        for _ix in rt_index]
            # matches
            for seq, rt in zip(seqs, rt_index):
                ix = [i + 1 for i, r in enumerate(seq) if r == _t]
                adds += [SeqCorrect(rt_idx=rt, mods=[ModSite(mass, i, _m)])
                         for i in ix]

        return adds

    def _combine_mods(self, seq, mods):
        """ Insert modification after target residue. """
        if not mods:
            return seq

        # separate modifications
        term_mods, int_mods = [""] * 2, []
        for mod in mods:
            if isinstance(mod.site, str):
                term_mods[0 if mod.site == "nterm" else 1] = f"[{mod.mod}]"
            else:
                int_mods.append(mod)
        int_mods = sorted(int_mods, key=attrgetter("site"))

        # combine modifications and sequences
        frags, i = [], 0
        for mod in int_mods:
            frags.append(f"{seq[i:mod.site]}[{mod.mod}]")
            i = mod.site
        if i < len(seq):
            frags.append(seq[i:])

        return "".join([term_mods[0], "".join(frags), term_mods[1]])

    def _reconstruct_peptide(self, parsed_match, seq_corr, mods_corr):
        """ Reconstruct the peptides after correction. """
        seq = parsed_match.seq_prefix + seq_corr + parsed_match.seq_suffix
        # re-construct modifications
        mods = parsed_match.mods_prefix.copy()
        npre = len(parsed_match.seq_prefix)
        for mod in mods_corr:
            mods.append(mod if isinstance(mod.site, str) else
                        mod._replace(site=npre+mod.site))
        # end modifications
        npre += len(seq_corr)
        for mod in parsed_match.mods_suffix:
            mods.append(mod._replace(site=npre+mod.site))

        return seq.upper(), mods

    def _sequence_tag_search(self, psm, syn_peptides):
        """ Search sequence tags and match to synthetic peptides. """
        # centroid spectrum
        _spectrum = psm.spectrum.centroid()
        # spectrum object
        proc_spectrum = _Spectrum(_spectrum._peaks, _spectrum.prec_mz)
        # denoise
        _ = proc_spectrum.denoise()
        # get sequence tags
        proc_spectrum.extract_sequence_tags()
        # if no tag is found, which means that the spectrum is bad, return None
        if not proc_spectrum.sequence_tags:
            return None

        # search the peptide using the tags
        matches = []
        for seq, mods in syn_peptides:
            prec_tags = proc_spectrum.match_sequence(seq, mods=mods)
            for tag in prec_tags:
                n = tag.length + 1
                ion_index = dict(
                    zip(("y", "b") if tag.ion_type == "y" else ("b", "y"),
                        (0, n))
                )
                # precursor mass of subsequence containing the tag
                ms = (sum(constants.AA_MASSES[a].mono for a in tag.seq_term)
                      + constants.FIXED_MASSES["H2O"]
                      + sum(mod.mass for mod in tag.mods_tag))
                mp = tag.mz_tag - constants.FIXED_MASSES["H"]
                # match object to tag precursor
                m = SynMatch(seq=tag.seq_term, mods=tag.mods_tag, charge=1,
                             num_ions=n, max_tag=n, ion_index=ion_index,
                             prec_mz=tag.mz_tag, delta_mass=[(ms - mp, 1)],
                             nterm_left=False, cterm_left=False)
                matches.append((m, tag))

        return matches

    def _tag_match_correct(self, correct_pep, tag):
        """ Correct the first match to full peptide sequence. """
        seq, mods = tag.pseq, tag.pmods
        n = len(correct_pep.seq)
        # correct precursor sequence using the corrected tag precursor
        if tag.ion_type == "y":
            corr_seq = seq[:tag.index] + correct_pep.seq
            ion_index = {"y": n, "b": 0}
            corr_mods = [mod for mod in mods if isinstance(mod.site, str)
                         or mod.site <= tag.index]
            for mod in correct_pep.mods:
                corr_mods.append(mod._replace(site=mod.site + int(tag.index)))
        else:
            corr_seq = correct_pep.seq + seq[tag.index+1:]
            ion_index = {"b": n, "y": 0}
            corr_mods = list(correct_pep.mods)
            for mod in mods:
                if isinstance(mod.site, int) and mod.site >= tag.index:
                    corr_mods.append(
                        mod._replace(site=mod.site - int(tag.index) + n)
                    )
                elif mod.site == "cterm":
                    corr_mods.append(mod)
        m = (sum(constants.AA_MASSES[a].mono for a in corr_seq)
             + constants.FIXED_MASSES["H2O"]
             + sum(mod.mass for mod in corr_mods))

        return SynMatch(seq=corr_seq, mods=corr_mods, max_tag=tag.length+1,
                        ion_index=ion_index, delta_mass=m)

    def _validate_candidates(self, psm, candidates):
        """ Validate candidates. """
        spectrum = psm.spectrum.centroid()
        # denoising, including deisotoping
        proc_spectrum = _Spectrum(spectrum._peaks, spectrum.prec_mz)
        denoised_spectrum = proc_spectrum.denoise()
        # prefilter candidates to get top 1000 matches
        candidates = self._prefiltering(candidates, denoised_spectrum)
        # new psms
        X = []
        for pep in candidates:
            _psm_c = PSM(psm.data_id, psm.spec_id, pep, spectrum)
            # do validation
            vk = featuregetter.extract_features(_psm_c)
            X.append([getattr(vk, _name) for _name in Feature_names])
        # validation scores
        scores = self.validator.predict(X, use_cv=False)

        # best match
        j = np.argmax(scores)
        best_psm = PSM(psm.data_id, psm.spec_id, candidates[j], spectrum)
        best_psm.site_score = scores[j]
        best_psm.ml_scores = self.validator.predict(X[j], use_cv=True)
        best_psm.features = featuregetter.extract_features(best_psm)

        return best_psm

    def _prefiltering(self, candidates, spectrum,
                      AAmass=constants.AA_MASSES, tol=0.2):
        """
        Filter candidates prior to validation. The best 1000 candidates
        will be returned for further validation.
        """
        if len(candidates) <= 1000:
            return candidates

        mz, n = np.sort(spectrum[:, 0]), spectrum.shape[0]
        mh, mh2o = constants.FIXED_MASSES["H"], constants.FIXED_MASSES["H2O"]
        # quick annotations simply using b and y ions.
        num_ions = []
        for candidate in candidates:
            # get residue masses
            seq_mass = np.array([AAmass[a].mono for a in candidate.seq])
            # singly charged terminal adducts in [N-terminus, C-terminus]
            term_mass = [mh, mh2o + mh]
            for mod in candidate.mods:
                if isinstance(mod.site, int):
                    seq_mass[mod.site - 1] += mod.mass
                else:
                    term_mass[0 if mod.site == "nterm" else 1] += mod.mass
            # singly charged y and b ion m/z
            ybs = np.concatenate(
                (np.cumsum(seq_mass[:-1]) + term_mass[0],
                 np.cumsum(seq_mass[::-1][:-1]) + term_mass[1]),
                axis=0
            )
            ybs.sort()

            # do quick annotation
            mix = [bisect_left(mz, m) for m in ybs]
            nk = sum((k > 0 and m-mz[k-1] <= tol) or (k < n and mz[k]-m <= tol)
                     for k, m in zip(mix, ybs))
            num_ions.append(nk)

        # retain candidates with 1000 highest number of y and b ions annotated.
        sorted_ix = np.argsort(num_ions)[::-1]

        return [candidates[i] for i in sorted_ix[:1000]]

    def _restrict_refs(self, seq, mods, nterm=False, cterm=False):
        """
        Restrict residue combinations and artifacts according to seq.

        Parameters
        ----------
        seq: str
            The sequence for restricting the residue combinations.
        mods: List of ModSite object
            Modifications.

        Returns
        -------
        combs: dict
            Restricted combinations of residues.
        combs_add: dict
            Restricted combinations of residues as additional residues
            in permuations of synthesis error.
        artifacts: dict
            Artifacts.
        seq_arr: np.ndarray
            Sequence array.

        """
        nseq = len(seq)
        # get residue mass w/o modifications
        seq_arr = np.array(list(seq), dtype=str)
        seq_mass = np.array([constants.AA_MASSES[a].mono for a in seq])
        seq_mass_mod = seq_mass.copy()
        for mod in mods:
            if isinstance(mod.site, int):
                seq_arr[mod.site - 1] = seq_arr[mod.site - 1].lower()
                seq_mass_mod[mod.site - 1] += mod.mass

        # combinations for removing residues, modifications are considered
        combs = collections.defaultdict(dict)
        for i in range(6):
            res, mass = [], []
            for ix in itertools.combinations(range(nseq), i+1):
                res.append("".join(seq_arr[list(ix)]))
                mass.append(seq_mass_mod[list(ix)].sum())
            combs[i+1]["residues"] = np.array(res, dtype=str)
            combs[i+1]["mass"] = np.array(mass)

        # combinations for adding residues, modifications are excluded
        res = ["".join(seq[j] for j in ix) for i in range(2)
               for ix in itertools.combinations(range(nseq), i+1)]
        mass = np.array([seq_mass[list(ix)].sum() for i in range(2)
                         for ix in itertools.combinations(range(nseq), i+1)])
        combs_add = {"residues": res, "mass": mass}

        # remove artifacts if residues not existed in seq
        ix = [i for i, r in enumerate(self.artifacts["mods"][:, 1])
              if r in seq]
        if nterm:
            ix += np.where(self.artifacts["mods"][:, 1] == "nterm")[0].tolist()
        if cterm:
            ix += np.where(self.artifacts["mods"][:, 1] == "cterm")[0].tolist()
        artifacts = {key: vals[ix] for key, vals in self.artifacts.items()}

        return combs, combs_add, artifacts, seq_arr

    def _parse_matches(self, match):
        """
        Parse sequence into three regions according to tags:
        seq_prefix: first part
        seq_proc: middle part, which will be the target for error
                  correction.
        seq_suffix: the final part.
        """
        # mass of matched synthetic peptide
        seq, mods = match.seq, match.mods
        mass = [constants.AA_MASSES[a].mono for a in seq]

        # indices define subsequence for analysis
        nq = len(match.seq)
        if match.ion_index is not None:
            b, y = match.ion_index["b"], match.ion_index["y"]
            j0, j1 = (b, nq - y) if nq - y >= b else (nq - y, b)
            j0, j1 = max(j0 - 1, 0), min(nq, j1 + 2)
        else:
            j0, j1 = 0, nq

        # sequence and modifications for processing
        s0, s1, s2 = seq[:j0], seq[j0:j1], seq[j1:]
        m0, _, m2 = sum(mass[:j0]), sum(mass[j0:j1]), sum(mass[j1:])
        # separate modifications
        mod0, mod1, mod2 = [], [], []
        for mod in mods:
            if isinstance(mod.site, str) or mod.site <= j0:
                mod0.append(mod)
                m0 += mod.mass
            elif mod.site > j1:
                mod2.append(mod._replace(site=int(mod.site-j1)))
                m2 += mod.mass
            else:
                mod1.append(mod._replace(site=int(mod.site-j0)))

        return ProcMatch(seq_prefix=s0, mass_prefix=m0, mods_prefix=mod0,
                         seq_proc=s1, mods_proc=mod1, seq_suffix=s2,
                         mass_suffix=m2, mods_suffix=mod2)

    def _parse_raw_id(self, raw):
        """ Parse raw id. """
        raw_split = raw.split("_")
        j = [i for i, x in enumerate(raw_split) if x.endswith("mod")][0]
        target = "_".join(raw_split[j:j+2])
        if target in target_pairs:
            return target_pairs[target]
        return target

    def _get_targets(self):
        """ Get target information. """
        self.targets = json.load(open(r"ptm_experiment_info.json", "r"))

    def _get_synthetic_peptides(self):
        """ Load synthetic peptides. """
        # constants
        mo, mc = 15.994915, 57.021464
        # load synthetic peptides
        syn_peps = collections.defaultdict()
        # load unmodified peptides
        for target in self.targets.keys():
            name = self.targets[target]["unimod_name"]
            m = self.targets[target]["mass"]
            file_ = os.path.join(self.targets[target]["benchmark_path"],
                                 self.targets[target]["benchmark"])
            # load peptides
            mpeps = open(file_, "r").read().splitlines()
            benchmarks = []
            for pep in mpeps:
                # parse peptide to get modifications and sequences
                pep_sep = re.split("\[|\]", pep)
                seq, mods = "", []
                for i in range(1, len(pep_sep), 2):
                    seq += pep_sep[i - 1]
                    mods.append(ModSite(m, len(seq) if seq else "nterm", name))
                if i < len(pep_sep) - 1 or len(pep_sep) == 1:
                    seq += pep_sep[-1]

                # fixed modification at Cysteine
                if "C" in seq:
                    mods += [ModSite(mc, i + 1, "Carbamidomethyl")
                             for i, r in enumerate(seq) if r == "C"]
                    mods.sort(key=attrgetter("site"))
                benchmarks.append((seq, mods))

                # variable modification at Methionine
                _mets = [i + 1 for i, r in enumerate(seq) if r == "M"]
                for j in range(1, len(_mets) + 1):
                    for _ix in itertools.combinations(_mets, j):
                        mvar = [ModSite(mo, i, "Oxidation") for i in _ix]
                        benchmarks.append(
                            (seq, sorted(mods+mvar, key=attrgetter("site")))
                        )
            syn_peps[target] = benchmarks
        self.syn_peps = syn_peps

    def _get_artifact(self):
        """
        Get artifacts.

        Return
        ------
        dict
            mods: numpy array of modifications with sites
            mass: numpy array of mass of residue combinations

        """
        # artifacts
        with open(r"artifiacts.json", "r") as f:
            Artifacts = json.load(f)

        # parse the modifications
        _res, _mass = [], []
        for _mod in Artifacts.keys():
            for _site in Artifacts[_mod]["sites"]:
                _res.append([_mod, _site])
                _mass.append(Artifacts[_mod]["mass"])
        self.artifacts = {"mods": np.array(_res, dtype=str),
                          "mass": np.array(_mass)}

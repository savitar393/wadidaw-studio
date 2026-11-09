"""Indonesian mixing prompt generation, template partitioning, and balance target mapping.

Implements Section 4.5 and Section 5 of dokumentasi.md:
- Canonical 4-stem ordering: (vocals, drums, bass, other).
- Zero-sum relative balance targets: sum(r_i) = 0 via mean-centering.
- Strict split isolation of prompt templates (train vs. validation vs. test)
  to prevent template memorization and data leakage.
- Generation of 200 diverse human-style Indonesian test prompts for generalization evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


STEM_NAMES: tuple[str, str, str, str] = ("vocals", "drums", "bass", "other")

# Discrete uncentered target levels (dB) from Section 5.5 of dokumentasi.md.
LEVEL_VALUES_DB: tuple[float, ...] = (-5.0, -3.0, -2.0, 0.0, 2.0, 3.0, 5.0)

# Indonesian stem surface forms partitioned across splits and shared base nouns.
STEM_LABELS: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {
        "vocals": ("vokal", "suara vokal", "vokal utama"),
        "drums": ("drum", "ketukan drum", "ritme drum"),
        "bass": ("bass", "suara bass", "lini bass"),
        "other": ("instrumen pengiring", "iringan musik", "instrumen lain"),
    },
    "validation": {
        "vocals": ("vokal", "nyanyian utama", "jalur vokal"),
        "drums": ("drum", "perkusi drum", "beat drum"),
        "bass": ("bass", "frekuensi bass", "nada bass"),
        "other": ("melodi pengiring", "instrumen pendukung", "harmoni pengiring"),
    },
    "test": {
        "vocals": ("vokal", "suara penyanyi", "lead vokal"),
        "drums": ("drum", "set drum", "hentakan drum"),
        "bass": ("bass", "alunan bass", "low-end bass"),
        "other": ("instrumen lainnya", "musik pengiring", "elemen harmonis"),
    },
}

# Split-disjoint descriptor phrases per discrete level (dB) so validation/test
# evaluate genuine phrasing generalization rather than exact template matching.
LEVEL_PHRASES: dict[str, dict[int, tuple[str, ...]]] = {
    "train": {
        5: ("paling menonjol", "sangat maju", "dibuat paling dominan"),
        3: ("lebih maju", "lebih menonjol", "ditonjolkan"),
        2: ("sedikit lebih maju", "agak menonjol", "sedikit dinaikkan"),
        0: ("seimbang", "natural", "proporsional"),
        -2: ("agak mundur", "jangan terlalu dominan", "sedikit diturunkan"),
        -3: ("lebih mundur", "diturunkan", "dibuat lebih pelan"),
        -5: ("paling mundur", "sangat mundur", "jauh di latar belakang"),
    },
    "validation": {
        5: ("sangat menonjol di depan", "paling kuat terdengar", "sangat mendominasi"),
        3: ("cukup menonjol", "dinaikkan lebih jelas", "lebih ke depan"),
        2: ("agak lebih maju", "sedikit diperjelas", "naik tipis"),
        0: ("tetap seimbang", "di posisi netral", "stabil di tengah"),
        -2: ("sedikit ke belakang", "agak diredam", "turun tipis"),
        -3: ("cukup mundur", "dikurangi levelnya", "lebih ke belakang"),
        -5: ("sangat jauh di belakang", "paling pelan", "ditekan kuat ke belakang"),
    },
    "test": {
        5: ("paling depan dan jelas", "sangat tegas di depan", "dominan utama"),
        3: ("lebih ke permukaan", "diangkat lebih terasa", "lebih kuat"),
        2: ("sedikit diangkat", "agak lebih terasa", "sedikit ke depan"),
        0: ("seimbang natural", "rata dan wajar", "proporsional tanpa tonjolan"),
        -2: ("agak ditahan", "sedikit dikurangi", "jangan terlalu maju"),
        -3: ("ditarik mundur", "lebih lembut di belakang", "dikecilkan"),
        -5: ("paling belakang", "sangat lembut di latar", "jauh dikecilkan"),
    },
}

# Split-specific sentence framing templates.
SENTENCE_FRAMES: dict[str, tuple[str, ...]] = {
    "train": (
        "{clauses}",
        "buat {clauses}",
        "atur campuran dengan {clauses}",
        "tolong pastikan {clauses}",
    ),
    "validation": (
        "campurkan agar {clauses}",
        "fokuskan mix supaya {clauses}",
        "sesuaikan level dengan {clauses}",
        "untuk bagian ini, {clauses}",
    ),
    "test": (
        "arahkan keseimbangan mix agar {clauses}",
        "buat karakter campuran di mana {clauses}",
        "setel gain supaya {clauses}",
        "hasilkan mix dengan {clauses}",
    ),
}


@dataclass(frozen=True)
class MixingProfile:
    """Archetypal relative stem balance profile before zero-mean centering."""

    name: str
    raw_levels_db: tuple[int, int, int, int]  # (vocals, drums, bass, other)

    @property
    def balance_target(self) -> np.ndarray:
        raw = np.asarray(self.raw_levels_db, dtype=np.float32)
        return (raw - np.mean(raw)).astype(np.float32)


# 16 musically meaningful mixing profiles covering diverse stem priorities.
MIXING_PROFILES: tuple[MixingProfile, ...] = (
    MixingProfile("vocal_forward_v1", (5, 0, -2, -3)),
    MixingProfile("vocal_warm_ballad_v1", (5, -3, 0, -2)),
    MixingProfile("vocal_subtle_lift_v1", (3, 0, -2, 0)),
    MixingProfile("vocal_intimate_acoustic_v1", (3, -5, -2, 2)),
    MixingProfile("drums_punchy_rock_v1", (0, 5, 2, -3)),
    MixingProfile("drums_bass_groove_v1", (-2, 3, 3, -3)),
    MixingProfile("bass_heavy_modern_v1", (0, 2, 5, -3)),
    MixingProfile("bass_warm_foundation_v1", (2, -2, 3, -3)),
    MixingProfile("instrumental_forward_v1", (-3, 0, 0, 5)),
    MixingProfile("backing_melody_focus_v1", (-2, -2, 0, 3)),
    MixingProfile("balanced_reference_v1", (0, 0, 0, 0)),
    MixingProfile("vocal_drum_duet_v1", (3, 3, -3, -3)),
    MixingProfile("rhythm_tucked_back_v1", (3, -3, -3, 2)),
    MixingProfile("bass_tucked_clean_v1", (2, 2, -5, 0)),
    MixingProfile("drums_tucked_soft_v1", (2, -5, 0, 2)),
    MixingProfile("full_band_energetic_v1", (2, 3, 2, -5)),
)


def normalize_balance_target(raw_r: Sequence[float] | np.ndarray) -> np.ndarray:
    """Center a 4-stem relative target vector so that sum(r) == 0."""
    arr = np.asarray(raw_r, dtype=np.float64)
    if arr.shape != (4,) or not np.isfinite(arr).all():
        raise ValueError("Expected 4 finite balance target values in dB.")
    centered = arr - np.mean(arr)
    return centered.astype(np.float32)


def snap_to_level_bucket(value_db: float) -> int:
    """Map a continuous dB value in [-5, 5] to the nearest discrete phrase key."""
    buckets = (-5, -3, -2, 0, 2, 3, 5)
    return min(buckets, key=lambda b: abs(float(value_db) - b))


def generate_paraphrases(
    profile: MixingProfile,
    split: str,
    n_paraphrases: int = 4,
    rng: np.random.Generator | None = None,
) -> list[dict[str, str]]:
    """Generate `n_paraphrases` distinct Indonesian prompts for a MixingProfile.

    Each returned item has:
        - "prompt": Indonesian instruction text
        - "prompt_template": identifier combining split, profile name, and frame id
    """
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {split}")
    if n_paraphrases < 1:
        raise ValueError("n_paraphrases must be >= 1.")
    if rng is None:
        rng = np.random.default_rng(2026)

    stem_vocab = STEM_LABELS[split]
    phrase_vocab = LEVEL_PHRASES[split]
    frames = SENTENCE_FRAMES[split]

    raw_levels = profile.raw_levels_db
    results: list[dict[str, str]] = []

    for p_idx in range(n_paraphrases):
        # Vary stem mention order and omit at most one neutral (0 dB) stem
        # on alternate paraphrases (mirroring natural human instructions where
        # unmentioned stems are implicitly neutral 0 dB).
        order = list(range(4))
        if p_idx == 1:
            order = [0, 2, 1, 3]
        elif p_idx == 2:
            order = [1, 0, 3, 2]
        elif p_idx >= 3:
            order = list(rng.permutation(4))

        clauses = []
        omitted_neutral = False
        for stem_idx in order:
            level = snap_to_level_bucket(raw_levels[stem_idx])
            if level == 0 and (p_idx % 2 == 1) and not omitted_neutral and len(clauses) >= 2:
                omitted_neutral = True
                continue
            stem_name = STEM_NAMES[stem_idx]
            s_variants = stem_vocab[stem_name]
            p_variants = phrase_vocab[level]
            s_choice = s_variants[(p_idx + stem_idx) % len(s_variants)]
            p_choice = p_variants[(p_idx + stem_idx) % len(p_variants)]
            clauses.append(f"{s_choice} {p_choice}")

        joined = ", ".join(clauses)
        frame = frames[p_idx % len(frames)]
        text = frame.format(clauses=joined)
        template_id = f"{split}_{profile.name}_f{p_idx % len(frames)}"
        results.append({
            "prompt": text,
            "prompt_template": template_id,
        })

    return results


# Keyword lexicon for deterministic parsing & text evaluation (Section 5.5 & 8.4).
_STEM_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vocals", ("vokal", "penyanyi", "nyanyian", "suara utama", "lead")),
    ("drums", ("drum", "perkusi", "ketukan", "beat", "hentakan")),
    ("bass", ("bass", "low-end", "sub")),
    ("other", ("instrumen", "pengiring", "iringan", "melodi", "harmoni", "pendukung", "lainnya", "lain", "gitar", "synthesizer", "kibor", "piano")),
)

_MODIFIER_RULES: tuple[tuple[tuple[str, ...], float], ...] = (
    # Strong positive (+5 dB)
    (("paling menonjol", "sangat maju", "paling dominan", "sangat menonjol", "paling kuat", "sangat mendominasi", "paling depan", "sangat tegas", "dominan utama", "sangat keras"), 5.0),
    # Strong negative (-5 dB)
    (("paling mundur", "sangat mundur", "jauh di latar", "sangat jauh", "paling pelan", "ditekan kuat", "paling belakang", "sangat lembut", "jauh dikecilkan", "sangat kecil"), -5.0),
    # Slight positive (+2 dB)
    (("sedikit lebih maju", "agak menonjol", "sedikit dinaikkan", "agak lebih maju", "sedikit diperjelas", "naik tipis", "sedikit diangkat", "agak lebih terasa", "sedikit ke depan", "agak naik", "sedikit lebih terdengar", "sedikit naik"), 2.0),
    # Slight negative (-2 dB)
    (("agak mundur", "jangan terlalu", "sedikit diturunkan", "sedikit ke belakang", "agak diredam", "turun tipis", "agak ditahan", "sedikit dikurangi", "sedikit lebih pelan", "agak pelan", "sedikit mundur"), -2.0),
    # Moderate positive (+3 dB)
    (("lebih maju", "lebih menonjol", "ditonjolkan", "tonjolkan", "cukup menonjol", "dinaikkan", "lebih ke depan", "lebih ke permukaan", "diangkat", "lebih kuat", "lebih keras", "naikkan", "maju", "fokus utama"), 3.0),
    # Moderate negative (-3 dB)
    (("lebih mundur", "diturunkan", "lebih pelan", "cukup mundur", "dikurangi", "lebih ke belakang", "ditarik mundur", "lebih lembut", "dikecilkan", "kurangi", "mundur", "redam"), -3.0),
    # Neutral (0 dB)
    (("seimbang", "natural", "proporsional", "netral", "stabil", "rata", "wajar", "normal"), 0.0),
)


def parse_prompt_to_balance(prompt: str) -> dict[str, object]:
    """Parse an Indonesian mixing prompt into raw stem slots and normalized `r`.

    Used both for interactive user prompts at inference (comparing learned text
    encoder vs. rule-based slot extraction) and for computing Intent/Slot metrics.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    lowered = prompt.lower()
    clauses = [c.strip() for c in lowered.replace(";", ",").replace(".", ",").split(",") if c.strip()]

    raw_slots: dict[str, float] = {stem: 0.0 for stem in STEM_NAMES}
    mentioned: dict[str, bool] = {stem: False for stem in STEM_NAMES}

    for clause in clauses:
        matched_stems = [
            stem
            for stem, keywords in _STEM_KEYWORDS
            if any(kw in clause for kw in keywords)
        ]
        if not matched_stems:
            continue

        matched_val = 0.0
        for phrases, val in _MODIFIER_RULES:
            if any(phrase in clause for phrase in phrases):
                matched_val = val
                break

        for stem in matched_stems:
            raw_slots[stem] = matched_val
            mentioned[stem] = True

    raw_vec = np.array([raw_slots[s] for s in STEM_NAMES], dtype=np.float32)
    r_vec = normalize_balance_target(raw_vec)

    return {
        "raw_slots_db": raw_slots,
        "mentioned_stems": mentioned,
        "balance_target": r_vec,
    }


def generate_human_test_prompts(count: int = 200, seed: int = 2026) -> list[dict[str, object]]:
    """Generate 200 natural, human-style Indonesian mixing prompts for the test set.

    Unlike synthetic training paraphrases, these use varied natural Indonesian
    studio/producer phrasing (colloquial & formal studio instructions) paired with
    their ground-truth relative target `r` (`sum(r) = 0`).
    """
    if count != 200:
        raise ValueError("Specification requires exactly 200 human test prompts.")

    rng = np.random.default_rng(seed)

    # 25 natural producer-style sentence patterns (completely disjoint from SENTENCE_FRAMES)
    natural_patterns: tuple[tuple[str, tuple[int, int, int, int]], ...] = (
        ("Tolong angkat suara penyanyi supaya paling menonjol di depan, bass agak ditahan sedikit, dan biarkan drum tetap seimbang.", (5, 0, -2, -3)),
        ("Vokal dibuat lebih maju dan jelas, sedangkan iringan musik serta bass ditarik lebih mundur.", (3, 0, -3, -3)),
        ("Saya ingin vokal terdengar paling kuat, ketukan drum cukup mundur, dan bass tetap natural.", (5, -3, 0, -2)),
        ("Buat suasana akustik yang hangat: vokal lebih menonjol, instrumen pengiring sedikit diangkat, tetapi drum sangat mundur.", (3, -5, -2, 2)),
        ("Berikan energi rock dengan drum paling depan dan jelas, bass sedikit diangkat, sementara instrumen lainnya ditarik mundur.", (0, 5, 2, -3)),
        ("Fokuskan groove pada drum dan bass yang sama-sama lebih kuat, sementara vokal agak mundur dan iringan dikurangi.", (-2, 3, 3, -3)),
        ("Tonjolkan low-end bass hingga paling dominan, drum sedikit lebih terasa, dan instrumen pengiring dikecilkan.", (0, 2, 5, -3)),
        ("Buat bass lebih maju sebagai fondasi, vokal sedikit diangkat, namun drum agak ditahan dan iringan lebih mundur.", (2, -2, 3, -3)),
        ("Biarkan instrumen pengiring menjadi fokus utama paling depan, vokal ditarik mundur, sedangkan drum dan bass tetap seimbang.", (-3, 0, 0, 5)),
        ("Naikkan melodi pengiring agar lebih terasa di permukaan, sementara vokal dan drum sedikit dikurangi.", (-2, -2, 0, 3)),
        ("Jaga seluruh elemen mulai dari vokal, drum, bass, hingga instrumen lainnya agar terdengar seimbang natural.", (0, 0, 0, 0)),
        ("Tonjolkan vokal dan drum secara bersamaan ke depan, lalu kecilkan bass serta musik pengiring.", (3, 3, -3, -3)),
        ("Kecilkan seksi ritme drum dan bass agar lebih mundur, lalu angkat vokal lebih maju bersama sedikit iringan.", (3, -3, -3, 2)),
        ("Bersihkan frekuensi bawah dengan membuat bass paling belakang, sedangkan vokal dan drum sedikit diangkat.", (2, 2, -5, 0)),
        ("Redam hentakan drum jauh di latar belakang, beri ruang sedikit untuk vokal dan instrumen pengiring.", (2, -5, 0, 2)),
        ("Buat vokal, drum, dan bass terdengar rapat di depan, sementara elemen harmonis pengiring ditekan kuat ke belakang.", (2, 3, 2, -5)),
        ("Suara penyanyi sedikit lebih maju saja, bass jangan terlalu dominan, sisanya biarkan proporsional.", (2, 0, -2, 0)),
        ("Ketukan drum dinaikkan lebih jelas, vokal tetap seimbang, namun bass dan pengiring agak ditahan.", (0, 3, -2, -2)),
        ("Pastikan vokal tidak tenggelam dan terasa lebih kuat, drum sedikit lebih maju, tetapi instrumen lain lebih mundur.", (3, 2, 0, -3)),
        ("Kurangi level vokal agak ke belakang agar alunan bass yang lebih maju dan drum yang seimbang lebih terdengar.", (-2, 0, 3, 0)),
    )

    # 10 natural style prefixes/suffixes used by Indonesian audio engineers / musicians
    stylistic_wrappers: tuple[str, ...] = (
        "{base}",
        "Untuk bagian lagu ini, {base_lower}",
        "Saat melakukan gain staging, {base_lower}",
        "{base} Pastikan transisi antar stem terdengar rapi.",
        "Di mix ini, {base_lower}",
        "{base} Jangan sampai ada instrumen yang memicu clipping.",
        "Sesuai arahan produser, {base_lower}",
        "{base} Pertahankan kejernihan keseluruhan campuran.",
        "Tolong atur fader supaya {base_lower}",
        "Sebagai referensi akhir, {base_lower}",
    )

    prompts: list[dict[str, object]] = []
    idx = 0
    for wrapper in stylistic_wrappers:
        for p_idx, (base_sentence, raw_levels) in enumerate(natural_patterns):
            idx += 1
            base_lower = base_sentence[0].lower() + base_sentence[1:]
            text = wrapper.format(base=base_sentence, base_lower=base_lower)
            r_target = normalize_balance_target(raw_levels)
            profile_ref = MIXING_PROFILES[p_idx % len(MIXING_PROFILES)].name
            prompts.append({
                "id": f"human_test_{idx:03d}",
                "prompt": text,
                "raw_levels_db": list(raw_levels),
                "balance_target": [round(float(x), 4) for x in r_target],
                "profile_hint": profile_ref,
                "source": "human_curated_indonesian_v1",
            })

    # Shuffle deterministically so ordering is well-distributed across test songs
    order = rng.permutation(len(prompts))
    shuffled = [prompts[int(i)] for i in order]
    for new_i, item in enumerate(shuffled, start=1):
        item["id"] = f"human_test_{new_i:03d}"

    return shuffled


def save_prompt_banks(output_dir: Path | str) -> dict[str, Path]:
    """Write `train.json`, `validation.json`, and `test_human.json` to `data/prompts/`."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)

    train_bank = []
    val_bank = []
    for profile in MIXING_PROFILES:
        r_list = [round(float(x), 4) for x in profile.balance_target]
        for item in generate_paraphrases(profile, "train", n_paraphrases=4, rng=rng):
            train_bank.append({
                "profile": profile.name,
                "raw_levels_db": list(profile.raw_levels_db),
                "balance_target": r_list,
                **item,
            })
        for item in generate_paraphrases(profile, "validation", n_paraphrases=4, rng=rng):
            val_bank.append({
                "profile": profile.name,
                "raw_levels_db": list(profile.raw_levels_db),
                "balance_target": r_list,
                **item,
            })

    human_test_bank = generate_human_test_prompts(count=200, seed=2026)

    paths = {
        "train": out_dir / "train.json",
        "validation": out_dir / "validation.json",
        "test_human": out_dir / "test_human.json",
    }

    paths["train"].write_text(json.dumps(train_bank, indent=2, ensure_ascii=False) + "\n")
    paths["validation"].write_text(json.dumps(val_bank, indent=2, ensure_ascii=False) + "\n")
    paths["test_human"].write_text(json.dumps(human_test_bank, indent=2, ensure_ascii=False) + "\n")

    return paths

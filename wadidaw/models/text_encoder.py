"""Text encoders for Indonesian mixing prompts (`384`-d sentence and token embeddings).

Implements Section 6.2 of dokumentasi.md:
- Supports `paraphrase-multilingual-MiniLM-L12-v2` (384-d) and `indobenchmark/indobert-base-p1`
  via `sentence-transformers` / `transformers` when installed (`backend="huggingface"`).
- Provides a deterministic, frozen `IndonesianSubwordTextEncoder` (`backend="subword"`, default
  offline-safe backend) combining:
    1. Syntactic clause segmentation (`split_mixing_clauses`) separating stem–modifier clauses,
    2. Compositional Role-Filler Tensor Product Binding (`s_c (x) m_c`, Smolensky 1990) binding
       Indonesian instrument role subspaces (`vocals`, `drums`, `bass`, `other`) with spatial/level
       modifier subspaces across clauses, and
    3. Character 3-gram subword embeddings for fine-grained lexical variation, projected via an
       orthogonal matrix into `R^384`.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wadidaw.data.prompts import split_mixing_clauses


TEXT_EMBED_DIM: int = 384
INDOBERT_EMBED_DIM: int = 768
INDOBERT_MODEL_NAME: str = "indobenchmark/indobert-base-p1"
_SUBWORD_DIM: int = 32


def get_text_embed_dim(backend: str) -> int:
    """Return the embedding width for a text backend."""
    return INDOBERT_EMBED_DIM if str(backend).lower() == "indobert" else TEXT_EMBED_DIM

_STEM_ROLE_LEXICON: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, ("vokal", "penyanyi", "nyanyian", "suara utama", "lead vokal", "jalur vokal", "suara vokal")),
    (1, ("drum", "perkusi", "ketukan", "beat", "hentakan", "set drum", "ritme drum")),
    (2, ("bass", "low-end", "sub", "frekuensi bawah", "alunan bass", "lini bass", "nada bass", "frekuensi bass")),
    (3, ("instrumen", "pengiring", "iringan", "melodi", "harmoni", "pendukung", "lainnya", "lain", "elemen harmonis", "gitar", "synthesizer", "kibor", "piano")),
)

# 7 modifier concept subspaces + continuous signed intensity scalar (total = 8 modifier features)
_MODIFIER_CONCEPT_LEXICON: tuple[tuple[int, float, tuple[str, ...]], ...] = (
    (0, 1.0, ("paling menonjol", "sangat maju", "paling dominan", "sangat menonjol", "paling kuat", "sangat mendominasi", "paling depan", "sangat tegas", "dominan utama", "fokus utama paling depan", "sangat keras")),
    (6, -1.0, ("paling mundur", "sangat mundur", "jauh di latar", "sangat jauh", "paling pelan", "ditekan kuat", "paling belakang", "sangat lembut", "jauh dikecilkan", "sangat kecil")),
    (2, 0.4, ("sedikit lebih maju", "agak menonjol", "sedikit dinaikkan", "agak lebih maju", "sedikit diperjelas", "naik tipis", "sedikit diangkat", "agak lebih terasa", "sedikit ke depan", "beri ruang sedikit", "sedikit lebih terasa", "agak naik", "sedikit lebih terdengar", "sedikit naik")),
    (4, -0.4, ("agak mundur", "jangan terlalu", "sedikit diturunkan", "sedikit ke belakang", "agak diredam", "turun tipis", "agak ditahan", "sedikit dikurangi", "agak ke belakang", "sedikit lebih pelan", "agak pelan", "sedikit mundur")),
    (1, 0.6, ("lebih maju", "lebih menonjol", "ditonjolkan", "tonjolkan", "cukup menonjol", "dinaikkan", "lebih ke depan", "lebih ke permukaan", "diangkat", "lebih terasa", "lebih kuat", "secara bersamaan ke depan", "rapat di depan", "angkat", "lebih keras", "naikkan", "maju", "fokus utama")),
    (5, -0.6, ("lebih mundur", "diturunkan", "lebih pelan", "cukup mundur", "dikurangi", "lebih ke belakang", "ditarik mundur", "lebih lembut", "dikecilkan", "kecilkan", "kurangi", "mundur", "redam")),
    (3, 0.0, ("seimbang", "natural", "proporsional", "netral", "stabil", "rata", "wajar", "normal")),
)


def _stable_hash(token: str, num_buckets: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, byteorder="little", signed=False)
    bucket = val % num_buckets
    sign = 1.0 if ((val >> 63) & 1) == 0 else -1.0
    return int(bucket), sign


class IndonesianSubwordTextEncoder(nn.Module):
    """Frozen 384-d compositional Role-Filler + Subword Indonesian text encoder."""

    def __init__(
        self,
        embed_dim: int = TEXT_EMBED_DIM,
        max_clauses: int = 8,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_clauses = max_clauses

        rng = np.random.default_rng(seed)

        self._ngram_buckets = 1024
        self._ngram_table = (
            rng.standard_normal((self._ngram_buckets, _SUBWORD_DIM), dtype=np.float32)
            / np.sqrt(_SUBWORD_DIM)
        )

        # Raw compositional feature vector per clause / sentence has dimension:
        # 32 (role-filler outer product 4x8) + 32 (mean-centered role-filler 4x8)
        # + 4 (stem presence) + 8 (modifier presence) + 32 (subword 3-gram) = 108
        raw_dim = 32 + 32 + 4 + 8 + _SUBWORD_DIM
        q_mat, _ = np.linalg.qr(rng.standard_normal(( max(embed_dim, raw_dim), max(embed_dim, raw_dim) )))
        self._proj_matrix = q_mat[:raw_dim, :embed_dim].astype(np.float32)

        pos_enc = rng.standard_normal((max_clauses, embed_dim), dtype=np.float32) * 0.02
        self.register_buffer("clause_pos", torch.from_numpy(pos_enc), persistent=False)

        self._sentence_cache: dict[str, np.ndarray] = {}
        self._token_cache: dict[str, np.ndarray] = {}

    def _subword_vec(self, text: str) -> np.ndarray:
        padded = f"<{text.lower().strip()}>"
        vec = np.zeros(_SUBWORD_DIM, dtype=np.float32)
        for i in range(max(0, len(padded) - 2)):
            b, sign = _stable_hash(padded[i : i + 3], self._ngram_buckets)
            vec += sign * self._ngram_table[b]
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec * 0.08  # Low weight so character noise never dominates role-filler semantics

    @staticmethod
    def _extract_clause_role_filler(clause: str) -> tuple[np.ndarray, np.ndarray]:
        s_vec = np.zeros(4, dtype=np.float32)
        for s_idx, patterns in _STEM_ROLE_LEXICON:
            if any(p in clause for p in patterns):
                s_vec[s_idx] = 1.0

        m_vec = np.zeros(8, dtype=np.float32)
        for b_idx, polarity, patterns in _MODIFIER_CONCEPT_LEXICON:
            if any(p in clause for p in patterns):
                m_vec[b_idx] = 1.0
                m_vec[7] = polarity
                break

        return s_vec, m_vec

    def _encode_single(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        if prompt in self._sentence_cache and prompt in self._token_cache:
            return self._sentence_cache[prompt], self._token_cache[prompt]

        clauses = split_mixing_clauses(prompt)
        if not clauses:
            clauses = [prompt.lower().strip()]
        clauses = clauses[: self.max_clauses]

        pos_np = self.clause_pos.cpu().numpy()
        clause_tokens = np.zeros((self.max_clauses, self.embed_dim), dtype=np.float32)

        rf_total = np.zeros((4, 8), dtype=np.float32)
        s_total = np.zeros(4, dtype=np.float32)
        m_total = np.zeros(8, dtype=np.float32)

        for c_idx, clause in enumerate(clauses):
            s_vec, m_vec = self._extract_clause_role_filler(clause)
            if s_vec.sum() > 0:
                rf_c = np.outer(s_vec, m_vec)
                # Overwrite earlier mention if same stem re-specified, otherwise accumulate
                for s_i in range(4):
                    if s_vec[s_i] > 0:
                        rf_total[s_i] = m_vec
                        s_total[s_i] = 1.0
                m_total += m_vec
            else:
                rf_c = np.zeros((4, 8), dtype=np.float32)

            rf_c_centered = rf_c - rf_c.mean(axis=0, keepdims=True)
            c_sub = self._subword_vec(clause)
            c_raw = np.concatenate(
                (rf_c.reshape(-1), rf_c_centered.reshape(-1), s_vec * 0.25, m_vec * 0.25, c_sub),
                axis=0,
            )
            c_proj = c_raw @ self._proj_matrix
            c_norm = float(np.linalg.norm(c_proj))
            if c_norm > 1e-8:
                c_proj = c_proj / c_norm
            clause_tokens[c_idx] = c_proj + pos_np[c_idx]

        # Sentence-level Role-Filler binding with zero-mean relational centering across stems
        rf_sent_centered = rf_total - rf_total.mean(axis=0, keepdims=True)
        sent_sub = self._subword_vec(prompt)
        sent_raw = np.concatenate(
            (
                rf_total.reshape(-1),
                rf_sent_centered.reshape(-1) * 1.5,
                s_total * 0.2,
                m_total * 0.2,
                sent_sub,
            ),
            axis=0,
        )
        sent_vec = sent_raw @ self._proj_matrix
        s_norm = float(np.linalg.norm(sent_vec))
        if s_norm > 1e-8:
            sent_vec = sent_vec / s_norm

        self._sentence_cache[prompt] = sent_vec.astype(np.float32)
        self._token_cache[prompt] = clause_tokens.astype(np.float32)
        return self._sentence_cache[prompt], self._token_cache[prompt]

    @torch.no_grad()
    def encode_sentences(self, prompts: Sequence[str], device: torch.device | str = "cpu") -> torch.Tensor:
        vecs = [self._encode_single(str(p))[0] for p in prompts]
        return torch.from_numpy(np.stack(vecs, axis=0)).to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def encode_tokens(self, prompts: Sequence[str], device: torch.device | str = "cpu") -> torch.Tensor:
        toks = [self._encode_single(str(p))[1] for p in prompts]
        return torch.from_numpy(np.stack(toks, axis=0)).to(device=device, dtype=torch.float32)

    def forward(self, prompts: Sequence[str] | torch.Tensor) -> torch.Tensor:
        if isinstance(prompts, torch.Tensor):
            return prompts
        return self.encode_sentences(prompts, device=self.clause_pos.device)


class TextEncoderWrapper(nn.Module):
    """Unified frozen text encoder.

    Backends:
        - ``subword``: deterministic offline Indonesian encoder (default).
        - ``huggingface``: SentenceTransformer sentence embeddings.
        - ``indobert``: IndoBERT token + sentence embeddings (frozen).
    """

    def __init__(
        self,
        backend: str = "subword",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embed_dim: int = TEXT_EMBED_DIM,
        max_clauses: int = 8,
    ) -> None:
        super().__init__()
        self.backend = str(backend).lower()
        self.model_name = INDOBERT_MODEL_NAME if self.backend == "indobert" else model_name
        self.embed_dim = get_text_embed_dim(self.backend) if self.backend == "indobert" else embed_dim
        self.subword_encoder = IndonesianSubwordTextEncoder(
            embed_dim=embed_dim,
            max_clauses=max_clauses,
        )
        self._hf_model = None
        self._hf_tokenizer = None
        self._indobert_model = None

        if self.backend == "huggingface":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._hf_model = SentenceTransformer(self.model_name)
                for param in self._hf_model.parameters():
                    param.requires_grad_(False)
            except Exception as exc:
                raise RuntimeError(
                    f"HuggingFace sentence-transformers backend requested ({self.model_name}) "
                    f"but unavailable: {exc}. Use backend='subword' for offline execution."
                ) from exc

        if self.backend == "indobert":
            try:
                from transformers import AutoModel, AutoTokenizer  # type: ignore

                self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._indobert_model = AutoModel.from_pretrained(self.model_name)
                self.embed_dim = int(self._indobert_model.config.hidden_size)
                for param in self._indobert_model.parameters():
                    param.requires_grad_(False)
            except Exception as exc:
                raise RuntimeError(
                    f"IndoBERT backend requested ({self.model_name}) but unavailable: {exc}. "
                    "Install `transformers` and ensure the model can be downloaded."
                ) from exc

    @torch.no_grad()
    def _encode_indobert(
        self,
        prompts: Sequence[str],
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._hf_tokenizer is not None
        assert self._indobert_model is not None
        encoded = self._hf_tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self._indobert_model(**encoded)
        tokens = outputs.last_hidden_state.to(dtype=torch.float32)
        mask = encoded["attention_mask"].to(dtype=torch.bool)
        return tokens, mask

    @torch.no_grad()
    def encode_sentences(self, prompts: Sequence[str] | torch.Tensor, device: torch.device | str) -> torch.Tensor:
        if isinstance(prompts, torch.Tensor):
            return prompts.to(device=device, dtype=torch.float32)
        if self.backend == "indobert":
            tokens, mask = self._encode_indobert(prompts, device=device)
            mask_f = mask.unsqueeze(-1).to(dtype=tokens.dtype)
            pooled = (tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            return F.normalize(pooled, dim=-1)
        if self._hf_model is not None:
            emb = self._hf_model.encode(
                list(prompts),
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return emb.to(device=device, dtype=torch.float32)
        return self.subword_encoder.encode_sentences(prompts, device=device)

    @torch.no_grad()
    def encode_tokens(self, prompts: Sequence[str] | torch.Tensor, device: torch.device | str) -> torch.Tensor:
        if isinstance(prompts, torch.Tensor) and prompts.ndim == 3:
            return prompts.to(device=device, dtype=torch.float32)
        if self.backend == "indobert":
            tokens, _mask = self._encode_indobert(prompts, device=device)
            return tokens
        return self.subword_encoder.encode_tokens(prompts, device=device)

    @torch.no_grad()
    def encode_tokens_with_mask(
        self,
        prompts: Sequence[str] | torch.Tensor,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return token embeddings and a boolean mask for valid tokens."""
        if isinstance(prompts, torch.Tensor) and prompts.ndim == 3:
            tokens = prompts.to(device=device, dtype=torch.float32)
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=device)
            return tokens, mask
        if self.backend == "indobert":
            return self._encode_indobert(prompts, device=device)
        tokens = self.subword_encoder.encode_tokens(prompts, device=device)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=device)
        return tokens, mask

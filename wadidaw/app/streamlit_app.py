"""Streamlit application for Text-Guided Automatic Gain Staging on MUSDB18-HQ stems.

Implements Section 9 of dokumentasi.md:
- Upload 4 stems (`vocals.wav`, `drums.wav`, `bass.wav`, `other.wav`) or load a test track from MUSDB18-HQ.
- Input Indonesian mixing prompt + quick-select presets.
- Choose a trained model checkpoint (`M3`, `M1`, `M2`, `M4`) with automatic CPU fallback.
- Display predicted per-stem gains (dB), stem loudnesses, and safety metrics (`peak_dbfs`, `clipping_rate`).
- Preview and download `final_mix.wav`.
"""

from __future__ import annotations

import io
from pathlib import Path

import soundfile as sf

from wadidaw.app.inference import CHECKPOINT_DIR, predict_and_render_mix
from wadidaw.data.musdb import MUSDB_ROOT
from wadidaw.data.prompts import STEM_NAMES


EXAMPLE_PROMPTS: tuple[str, ...] = (
    "vokal paling menonjol, bass agak mundur, drum seimbang",
    "buat drum lebih maju, bass sedikit lebih maju, instrumen pengiring lebih mundur",
    "suara penyanyi lebih maju dan hangat, ketukan drum cukup mundur, bass tetap seimbang",
    "instrumen pengiring paling menonjol di depan, vokal agak mundur, drum dan bass seimbang",
    "jaga seluruh elemen vokal, drum, bass, dan instrumen pengiring agar seimbang natural",
)


def list_available_checkpoints(checkpoint_dir: Path = CHECKPOINT_DIR) -> list[Path]:
    if not checkpoint_dir.is_dir():
        return []
    return sorted(checkpoint_dir.glob("*_best.pt"))


def list_demo_test_songs(musdb_root: Path = MUSDB_ROOT) -> list[Path]:
    test_dir = musdb_root / "test"
    if not test_dir.is_dir():
        return []
    return sorted(p for p in test_dir.iterdir() if p.is_dir())


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="WadiDAW Studio — Text-Guided Automatic Gain Staging",
        layout="wide",
    )
    st.title("WadiDAW Studio: Text-Guided Automatic Gain Staging")
    st.caption(
        "Sistem Pencampuran Musik Berbasis Instruksi Teks Bahasa Indonesia "
        "Menggunakan Model Multimodal Audio–Teks (MUSDB18-HQ)"
    )

    with st.sidebar:
        st.header("Konfigurasi Model")
        ckpts = list_available_checkpoints()
        ckpt_options = {p.stem.replace("_best", ""): p for p in ckpts}
        ckpt_labels = list(ckpt_options.keys()) + ["DSP + Rule Parser (Fallback)"]
        default_idx = 0
        for idx, label in enumerate(ckpt_labels):
            if label.startswith("m3"):
                default_idx = idx
                break

        selected_label = st.selectbox("Pilih Model Checkpoint", ckpt_labels, index=default_idx)
        selected_ckpt = ckpt_options.get(selected_label)

        input_mode = st.radio(
            "Sumber Audio 4 Stem",
            ("Upload 4 File WAV", "Pilih Lagu Test MUSDB18-HQ"),
        )
        render_duration = st.slider(
            "Durasi Render Preview (detik)",
            min_value=8,
            max_value=60,
            value=16,
            step=4,
        )

    col_left, col_right = st.columns([1, 1])

    stem_sources: dict[str, object] = {}
    start_sec = 0.0

    with col_left:
        st.subheader("1. Input 4 Stem Audio")
        if input_mode == "Upload 4 File WAV":
            for stem_name in STEM_NAMES:
                uploaded = st.file_uploader(
                    f"Upload `{stem_name}.wav`",
                    type=["wav", "flac"],
                    key=f"upload_{stem_name}",
                )
                if uploaded is not None:
                    stem_sources[stem_name] = io.BytesIO(uploaded.getvalue())
        else:
            test_songs = list_demo_test_songs()
            if not test_songs:
                st.warning("Folder `data/raw/musdb18hq/test` tidak ditemukan.")
            else:
                song_map = {p.name: p for p in test_songs}
                chosen_song = st.selectbox("Pilih Lagu dari Test Set MUSDB18-HQ", list(song_map.keys()))
                start_sec = float(
                    st.slider("Detik Mulai Excerpt Analisis", min_value=0, max_value=60, value=24, step=4)
                )
                song_dir = song_map[chosen_song]
                for stem_name in STEM_NAMES:
                    stem_sources[stem_name] = song_dir / f"{stem_name}.wav"

    with col_right:
        st.subheader("2. Instruksi Prompt Bahasa Indonesia")
        preset = st.selectbox("Contoh Prompt Cepat", ("(Ketik manual)", *EXAMPLE_PROMPTS))
        default_prompt = EXAMPLE_PROMPTS[0] if preset == "(Ketik manual)" else preset
        prompt_text = st.text_area(
            "Masukkan Instruksi Mixing:",
            value=default_prompt,
            height=100,
        )

        mix_clicked = st.button("Mix 4 Stem", type="primary", use_container_width=True)

    if mix_clicked:
        missing = [s for s in STEM_NAMES if s not in stem_sources]
        if missing:
            st.error(f"File stem belum lengkap! Stem yang belum diunggah: {', '.join(missing)}")
            return
        if not prompt_text.strip():
            st.error("Prompt instruksi teks tidak boleh kosong.")
            return

        with st.spinner("Menganalisis log-mel 4 stem dan menghitung gain optimal..."):
            try:
                result = predict_and_render_mix(
                    stem_sources=stem_sources,
                    prompt=prompt_text,
                    checkpoint_path=selected_ckpt,
                    device="cpu",
                    analysis_start_sec=start_sec,
                    max_render_duration_sec=float(render_duration),
                )
            except Exception as exc:
                st.error(f"Gagal memproses audio: {exc}")
                return

        st.divider()
        st.subheader("3. Hasil Prediksi Gain & Render Stereo Mix")

        g_cols = st.columns(4)
        for idx, stem_name in enumerate(STEM_NAMES):
            gain_val = result["gains_db"][stem_name]
            loud_val = result["loudness_measured_db"][stem_name]
            is_act = result["active_mask"][stem_name]
            status_txt = f"Loudness awal: {loud_val:+.1f} dBFS" if is_act else "Silent (di-mask 0 dB)"
            g_cols[idx].metric(
                label=f"Gain {stem_name.upper()}",
                value=f"{gain_val:+.2f} dB",
                delta=status_txt,
            )

        m_col1, m_col2, m_col3 = st.columns(3)
        metrics = result["metrics"]
        m_col1.metric("Peak Level", f"{metrics['peak_dbfs']:.2f} dBFS")
        m_col2.metric("Clipping Rate", f"{metrics['clipping_rate'] * 100:.2f}%")
        m_col3.metric("Model Aktif", str(result["model_used"]).upper())

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, result["mix"], result["sample_rate"], format="WAV", subtype="PCM_16")
        wav_bytes = wav_buffer.getvalue()

        st.audio(wav_bytes, format="audio/wav")
        st.download_button(
            label="Download final_mix.wav (44.1 kHz Stereo)",
            data=wav_bytes,
            file_name="final_mix.wav",
            mime="audio/wav",
        )


if __name__ == "__main__":
    run_streamlit_app()

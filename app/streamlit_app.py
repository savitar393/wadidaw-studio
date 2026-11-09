"""Entrypoint wrapper so `streamlit run app/streamlit_app.py` works from the repo root."""

from wadidaw.app.streamlit_app import run_streamlit_app

if __name__ == "__main__":
    run_streamlit_app()

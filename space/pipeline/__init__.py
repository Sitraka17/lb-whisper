"""Local meeting-notes pipeline: transcribe → diarize → merge → summarize → export.

Heavy dependencies (faster-whisper, pyannote, llama-cpp) are imported inside functions so
that merge.py, export.py and the extractive summariser stay importable without them.
"""

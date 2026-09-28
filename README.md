# TrustLens: A Multimodal AI System for Detecting and Explaining Phishing and Scam Attempts

**Module:** CM3070 Final Project (University of London) · **Template:** CM3020 Artificial Intelligence, Project Idea 1: *Orchestrating AI models to achieve a goal*

TrustLens helps non-specialist users check a suspicious **text message, screenshot or voice note**.
Every input is reduced to text and passed through one shared pipeline that combines pre-trained
AI models with transparent security checks. The result is a **risk level**, the **reasons** in plain
language, the **evidence** behind them and **recommended next steps**.

Everything runs locally: messages, screenshots and audio are processed in memory and never stored.

---

## Orchestrated models

| # | Module | Pre-trained model | Data space | Question it answers |
|---|--------|-------------------|------------|---------------------|
| 1 | `classifier.py` | `ealvaradob/bert-finetuned-phishing` (BERT-large) | Text | Is this phishing? |
| 2 | `emotion.py` | `j-hartmann/emotion-english-distilroberta-base` | Text | Is it applying emotional pressure? |
| 3 | `ocr.py` | Tesseract OCR | Image | What text is in this screenshot? |
| 4 | `speech.py` | OpenAI Whisper (base) | Audio | What was said in this voice note? |

## Transparent evidence and fusion

| Module | Role |
|--------|------|
| `indicators.py` | Rule-based warning signs (urgency, credential requests, payment pressure and more), each with its own explanation |
| `link_analysis.py` | URL forensics: IP-address hosts, deceptive sub-domains, punycode, shorteners, abused domain endings, brand/domain mismatch |
| `context_signals.py` | Evidence that a message is genuine (order and tracking numbers, unsubscribe footers, known-good domains) |
| `calibration.py` | Temperature scaling for the over-confident classifier, plus ECE, Brier score and temperature fitting |
| `aggregator.py` | Risk fusion (default and calibrated paths) and templated explanations that cannot invent advice |
| `attribution.py` | Optional word-level explanation by leave-one-out occlusion (display only; never changes the score) |
| `orchestrator.py` | Reduce-to-text routing: screenshots and voice notes are converted to text, then share one pipeline |

## Project structure

```
app.py                  FastAPI backend and single-page web interface
orchestrator.py         Shared analysis pipeline
classifier.py           Model 1: phishing classifier
emotion.py              Model 2: emotional-pressure score
ocr.py                  Model 3: Tesseract OCR
speech.py               Model 4: Whisper transcription
indicators.py           Rule-based warning signs
link_analysis.py        URL forensics
context_signals.py      Benign-context evidence
calibration.py          Temperature scaling and calibration metrics
aggregator.py           Risk fusion and explanations
attribution.py          Word attribution (occlusion)
trustlens_cli.py        Command-line scanner
evaluate.py             End-to-end metrics on the 25-message test set
evaluate_models.py      Model-specific and calibration analysis (preliminary report)
evaluate_chapter5.py    Final evaluation used in Chapter 5 (live models, OCR, Whisper, timing)
analyse_user_study.py   SUS and user-study analysis from the Google Form export
make_figures.py         Regenerates the preliminary-report figures
make_architecture.py    Draws the early architecture figure (assets/)
evaluation_results.csv  Recorded scores for the 25 test messages (used by the regression test)
eval_outputs/           CSV results and figures produced by evaluate_chapter5.py
eval_inputs/voice/      references.csv for the recorded voice notes
tests/                  pytest suite (11 tests)
assets/                 Figures from the preliminary report
```

## Download

Clone the repository, or click **Code → Download ZIP** on GitHub and unzip it:

```bash
git clone https://github.com/samiksha756/TrustLens.git
cd TrustLens
```

You need **Python 3.12** installed. The AI models download automatically the first time
they are used: the two text models from Hugging Face (about 1.6 GB in total) on the first
analysis, and Whisper base (about 140 MB) on the first voice note. An internet connection
is needed for the first run only.

## Setup

Tested with **Python 3.12 on macOS (Intel x86_64)**.

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt     # the text models download on first use
```

The screenshot and voice routes also need two system tools:

| Tool | Needed for | Install |
|------|------------|---------|
| `tesseract` | Screenshots (OCR) | `brew install tesseract`, or `conda install -c conda-forge tesseract` |
| `ffmpeg` | Voice notes (Whisper) | `brew install ffmpeg`, or `conda install -c conda-forge ffmpeg` |

On Intel Macs where Homebrew no longer provides prebuilt packages, installing both tools from
conda-forge into a separate environment and adding them to your `PATH` works. The version pins
in `requirements.txt` (NumPy below 2, numba 0.60.0, llvmlite 0.43.0) avoid compiling from source.

If either tool is missing, TrustLens still runs: that route reports that it is unavailable and
the text route keeps working.

## Run

```bash
python -m uvicorn app:app --port 8000
```

Then open <http://127.0.0.1:8000>. The first analysis takes about 30 seconds while the models load.

### Command line

```bash
python trustlens_cli.py "Your account is locked, verify at http://paypal.com.secure-login.tk/auth"
python trustlens_cli.py --calibrated --file messages.txt
python trustlens_cli.py --json "..."
```

The exit code is 0 for Safe/Caution and 2 for Suspicious/High Risk.

## Tests and evaluation

```bash
pytest -v                                  # 11 tests; the suite runs without model weights
python evaluate_chapter5.py --quick        # final evaluation; results in eval_outputs/
python analyse_user_study.py responses.csv # user-study analysis (Google Form CSV export)
```

For the voice-note evaluation, place your recordings in `eval_inputs/voice/` next to
`references.csv`, which lists each file name and exactly what was said.

## Key results

| Measure | Result |
|---------|--------|
| Detection, calibrated path (25 messages) | Recall 1.00, false-positive rate 0.00 |
| Detection, default path | Recall 1.00, false-positive rate 0.17 |
| OCR on rendered messages | Mean character error rate 0.2% |
| Whisper on recorded voice notes | Mean word error rate 7.9% |
| Screenshot and voice routes | Same risk level as the typed text for every input |
| User study (5 participants) | Correct judgements 65% → 100%; mean SUS 65.5 |
| Lighthouse accessibility audit | 100 / 100 |

## Limitations

The test set is small and self-written, OCR was tested on clean rendered images, and the voice
notes came from one speaker in a quiet room. TrustLens supports English only and is a
decision-support prototype, not a guarantee that a message is safe or malicious.

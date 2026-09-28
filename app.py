"""
TrustLens — FastAPI Backend (multimodal)
----------------------------------------
Exposes three analysis routes that all feed the SAME orchestrated pipeline:

    POST /api/analyse         { "text": "...", "calibrate": false, "explain": false }
    POST /api/analyse-image   multipart file "image"  (?calibrate=&explain=)
    POST /api/analyse-audio   multipart file "audio"  (?calibrate=&explain=)

`calibrate` is optional and defaults to false, so the default behaviour and all
previously reported numbers are preserved. When true, the calibrated,
link-aware, benign-context-aware scoring path is used.

`explain` is optional and defaults to false. When true, the response also
carries opt-in occlusion-based token attribution ("why these words?"). It is
slower (one classifier pass per word) so it is only computed on request.

Run with:
    python -m uvicorn app:app --reload --port 8000
Then open http://localhost:8000
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from orchestrator import analyse_text, analyse_image, analyse_audio

app = FastAPI(title="TrustLens", description="Multimodal phishing & scam risk analysis")


class AnalyseRequest(BaseModel):
    text: str
    calibrate: bool = False
    explain: bool = False


@app.post("/api/analyse")
def analyse(req: AnalyseRequest):
    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "Empty input"}, status_code=400)
    return analyse_text(text, source="text", calibrate=req.calibrate,
                        explain=req.explain)


@app.post("/api/analyse-image")
async def analyse_image_route(image: UploadFile = File(...),
                              calibrate: bool = False, explain: bool = False):
    data = await image.read()
    result = analyse_image(data, calibrate=calibrate, explain=explain)
    if "error" in result and "assessment" not in result:
        return JSONResponse(result, status_code=400)
    return result


@app.post("/api/analyse-audio")
async def analyse_audio_route(audio: UploadFile = File(...),
                              calibrate: bool = False, explain: bool = False):
    data = await audio.read()
    suffix = "." + (audio.filename.rsplit(".", 1)[-1] if "." in audio.filename else "wav")
    result = analyse_audio(data, suffix=suffix, calibrate=calibrate, explain=explain)
    if "error" in result and "assessment" not in result:
        return JSONResponse(result, status_code=400)
    return result


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>TrustLens — Multimodal Phishing &amp; Scam Risk Analyser</title>
     <link rel="icon" href="data:,">
  <style>
    :root {
      --blue:#1e51c9; --blue-dark:#163c95; --ink:#1f2430; --muted:#5a6472;
      --line:#c9ced8; --bg:#f4f6fb; --card:#ffffff; --focus:#0b3ea8;
    }
    * { box-sizing: border-box; }
    html { -webkit-text-size-adjust: 100%; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: var(--bg); margin: 0; padding: 2rem 1.25rem; color: var(--ink);
           line-height: 1.55; }
    .container { max-width: 780px; margin: 0 auto; }
    h1 { font-size: 1.8rem; margin: 0 0 0.25rem 0; }
    .subtitle { color: var(--muted); margin: 0 0 1.5rem 0; }

    /* Skip link for keyboard / screen-reader users */
    .skip-link { position:absolute; left:-999px; top:0; background:var(--blue); color:#fff;
                 padding:0.5rem 0.9rem; border-radius:0 0 6px 0; z-index:10; }
    .skip-link:focus { left:0; }

    /* Visible focus for every interactive element */
    a:focus-visible, button:focus-visible, [role="tab"]:focus-visible,
    input:focus-visible, textarea:focus-visible {
      outline: 3px solid var(--focus); outline-offset: 2px;
    }
    .sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px;
               overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; border:0; }

    .tabs { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
    [role="tab"] { padding: 0.5rem 1rem; border: 1px solid var(--line); border-radius: 6px;
           background: #fff; color: var(--ink); cursor: pointer; font-size: 0.95rem; }
    [role="tab"][aria-selected="true"] { background: var(--blue); color: #fff; border-color: var(--blue); }

    textarea { width: 100%; min-height: 150px; padding: 0.85rem; border: 1px solid var(--line);
               border-radius: 8px; font-size: 1rem; font-family: inherit; resize: vertical; }
    label { display: inline-block; }
    .field-label { font-weight: 600; display:block; margin-bottom: 0.35rem; }
    input[type=file] { margin: 0.5rem 0; font-size: 0.95rem; max-width: 100%; }
    button.action { margin-top: 0.75rem; padding: 0.7rem 1.5rem; background: var(--blue); color: #fff;
             border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; }
    button.action:hover { background: var(--blue-dark); }
    button.action:disabled { background: #8b93a3; cursor: not-allowed; }

    [role="tabpanel"][hidden] { display: none; }
    .toggles { margin-top: 0.8rem; display:flex; flex-direction:column; gap:0.5rem; }
    .toggle { font-size: 0.9rem; color: var(--muted); }
    .toggle input { margin-right: 0.4rem; }

    /* View switch (Simple / Detailed) */
    .viewswitch { display:flex; gap:0; margin:0 0 1rem 0; border:1px solid var(--line);
                  border-radius:8px; overflow:hidden; width:max-content; max-width:100%; }
    .viewswitch button { border:none; background:#fff; color:var(--ink); cursor:pointer;
                  padding:0.5rem 1rem; font-size:0.95rem; }
    .viewswitch button[aria-pressed="true"] { background:var(--blue); color:#fff; }

    /* Big plain-language bottom line (the at-a-glance verdict) */
    .bottomline { display:flex; align-items:flex-start; gap:0.6rem; padding:1rem 1.2rem;
                  border-radius:10px; margin-bottom:0.5rem; font-size:1.2rem; line-height:1.4;
                  font-weight:700; border:1px solid transparent; }
    .bottomline .bl-glyph { font-weight:900; font-size:1.35rem; line-height:1.2; }
    .bl-Safe { background:#e2f3e4; color:#14571b; border-color:#a5d4ab; }
    .bl-Caution { background:#fff3d9; color:#7a5300; border-color:#e8c98f; }
    .bl-Suspicious { background:#ffe6d6; color:#8a3b0a; border-color:#eab48f; }
    .bl-HighRisk { background:#fde1e0; color:#8a1a17; border-color:#e8a3a1; }

    /* Simple view hides the technical cards/lines (they return in Detailed view). */
    #result.simple-mode .detail { display:none !important; }

    @media (max-width: 600px) {
      .bottomline { font-size:1.05rem; }
      .viewswitch { width:100%; }
      .viewswitch button { flex:1 1 auto; }
    }

    .errorbox { display:none; margin-top:1rem; background:#fdecea; color:#7a1c14;
                border-left:4px solid #c62828; padding:0.75rem 1rem; border-radius:4px; }

    .result { margin-top: 2rem; display: none; }
    .badge { display: inline-flex; align-items:center; gap:0.4rem; padding: 0.4rem 0.9rem;
             border-radius: 999px; font-weight: 700; color: #fff; font-size: 0.98rem; }
    .badge .glyph { font-weight:900; }
    .badge-Safe { background: #256a2a; } .badge-Caution { background: #9a6b00; }
    .badge-Suspicious { background: #b0500a; } .badge-HighRisk { background: #a5201d; }

    .card { background: var(--card); border-radius: 10px; padding: 1.25rem 1.5rem; margin-top: 1rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08); border: 1px solid #e7eaf0; }
    .card h3 { margin-top: 0; font-size: 1.05rem; color: var(--ink); }
    .meter { height: 12px; background: #e4e8ef; border-radius: 999px; overflow: hidden;
             margin: 0.75rem 0 0.25rem 0; }
    .meter-fill { height: 100%; width: 0%; transition: width .4s ease; }

    .indicator, .link-finding, .ctx-finding { display: inline-block; padding: 0.25rem 0.6rem;
                 border-radius: 4px; margin: 0.15rem; font-size: 0.88rem; font-weight:600; }
    .indicator { background: #fff1dd; color: #8a4b00; border:1px solid #e8c48f; }
    .link-finding { background: #fde4e4; color: #8f1414; border:1px solid #e6a3a3; }
    .ctx-finding { background: #e2f3e4; color: #14571b; border:1px solid #a5d4ab; }
    .indicator-explain, .finding-explain { font-size: 0.9rem; color: #444b57; margin: 0.3rem 0 0.5rem 0; }

    /* Word-attribution ("why these words?") */
    .attn-legend { font-size:0.85rem; color:var(--muted); margin:0 0 0.6rem 0; }
    .attn-swatch { display:inline-block; width:0.8rem; height:0.8rem; border-radius:2px;
                   vertical-align:middle; margin:0 0.25rem 0 0.6rem; }
    .attn-row { display:flex; align-items:center; gap:0.6rem; margin:0.3rem 0; flex-wrap:wrap; }
    .attn-word { min-width:8rem; font-weight:600; }
    .attn-word.phishing { color:#8f1414; text-decoration: underline; text-decoration-style: solid; }
    .attn-word.safe { color:#14571b; text-decoration: underline; text-decoration-style: dotted; }
    .attn-bar { flex:1 1 8rem; height:12px; background:#e4e8ef; border-radius:999px; overflow:hidden; }
    .attn-bar-fill { height:100%; display:block; }
    .attn-bar-fill.phishing { background:#c0392b; }
    .attn-bar-fill.safe { background:#2e7d32; }
    .attn-dir { font-size:0.82rem; color:var(--muted); min-width:6.5rem; }

    .emotion-box { background: #fbe6ee; border-left: 4px solid #a31456; padding: 0.7rem 1rem;
                   border-radius: 4px; margin-top: 0.5rem; font-size: 0.95rem; color:#5a1030; }
    .action-box { background: #e5eefc; border-left: 4px solid var(--blue); padding: 0.9rem 1.1rem;
                  border-radius: 4px; margin-top: 0.75rem; color:#12305f; }
    .disclaimer { font-size: 0.85rem; color: #6b7280; margin-top: 1rem; font-style: italic; }
    .meta { font-size: 0.88rem; color: var(--muted); }
    .source-note { font-size: 0.85rem; color: var(--blue-dark); margin-top: 0.4rem; }

    /* ---------------------------------------------------------------
       Styles for the example buttons,
       the "Why" card, the "What should I do next?" choices and the
       "Check another message" button. No effect on analysis.
       --------------------------------------------------------------- */
    /* Secondary (outlined) buttons: examples, choices, check-another */
    button.secondary { margin-top: 0.75rem; padding: 0.6rem 1.1rem; background: #fff;
                       color: var(--blue-dark); border: 1px solid var(--blue); border-radius: 6px;
                       font-size: 0.95rem; cursor: pointer; }
    button.secondary:hover { background: #eef3fd; }
    .examples { display:flex; flex-wrap:wrap; gap:0.5rem; align-items:center; margin-top:0.25rem; }
    .examples-label { font-size:0.9rem; color:var(--muted); margin-top:0.75rem; }
    .examples button.secondary { margin-top: 0; }

    /* "Why" card: short plain-language reasons */
    .why-list { margin: 0.25rem 0 0 0; padding-left: 1.2rem; }
    .why-list li { margin: 0.3rem 0; }

    /* "What should I do next?" choice buttons + revealed steps */
    .choices { display:flex; flex-wrap:wrap; gap:0.5rem; }
    .choices button.secondary { margin-top: 0; }
    .choices button[aria-pressed="true"] { background: var(--blue); color:#fff; }
    .next-panel { margin-top: 0.9rem; }
    .next-panel ol { margin: 0.4rem 0 0 0; padding-left: 1.3rem; }
    .next-panel li { margin: 0.35rem 0; }
    .next-hint { font-size:0.9rem; color:var(--muted); margin:0.6rem 0 0 0; }

    /* Responsive: comfortable on phones */
    @media (max-width: 600px) {
      body { padding: 1rem 0.9rem; }
      h1 { font-size: 1.5rem; }
      .tabs { gap: 0.4rem; }
      [role="tab"] { flex: 1 1 auto; text-align:center; }
      button.action { width: 100%; }
      .choices button.secondary { flex: 1 1 100%; }   /* stack choices on phones */
      .card { padding: 1rem 1.1rem; }
      .attn-word { min-width: 6rem; }
    }
    /* Respect users who prefer reduced motion */
    @media (prefers-reduced-motion: reduce) {
      .meter-fill { transition: none; }
    }

  </style>
</head>
<body>
  <a href="#main" class="skip-link">Skip to content</a>
  <div class="container">
    <header>
      <h1><span aria-hidden="true">&#128269;</span> TrustLens</h1>
      <p class="subtitle">Multimodal phishing &amp; scam risk analysis — check a message, screenshot, or voice note.</p>
    </header>

    <main id="main">
      <div class="tabs" role="tablist" aria-label="Choose what to analyse">
        <button role="tab" id="tab-text" aria-selected="true" aria-controls="panel-text"
                data-panel="text" onclick="switchTab(this)">Text</button>
        <button role="tab" id="tab-image" aria-selected="false" aria-controls="panel-image"
                data-panel="image" tabindex="-1" onclick="switchTab(this)">Screenshot</button>
        <button role="tab" id="tab-audio" aria-selected="false" aria-controls="panel-audio"
                data-panel="audio" tabindex="-1" onclick="switchTab(this)">Voice note</button>
      </div>

      <div class="panel" role="tabpanel" id="panel-text" aria-labelledby="tab-text">
        <label class="field-label" for="input">Suspicious message</label>
        <textarea id="input" placeholder="Paste a suspicious email, SMS, or chat message here..."></textarea>
        <button class="action" id="btnText" onclick="analyseText()">Analyse text</button>

        <!-- "Try an example" buttons.
             Fill the box with a sample message and analyse it, so first-time
             users can see how TrustLens works without finding a scam first.
             Uses the normal analyse route; nothing special happens. -->
        <p class="examples-label">Not sure what to paste? Try an example:</p>
        <div class="examples">
          <button type="button" class="secondary" onclick="tryExample('scam')">Scam example</button>
          <button type="button" class="secondary" onclick="tryExample('safe')">Safe example</button>
        </div>
      </div>
      <div class="panel" role="tabpanel" id="panel-image" aria-labelledby="tab-image" hidden>
        <label class="field-label" for="imageFile">Screenshot of a suspicious message or web page (PNG/JPG)</label>
        <input type="file" id="imageFile" accept="image/*">
        <button class="action" id="btnImage" onclick="analyseImage()">Analyse screenshot</button>
      </div>
      <div class="panel" role="tabpanel" id="panel-audio" aria-labelledby="tab-audio" hidden>
        <label class="field-label" for="audioFile">Suspicious voice note (WAV/MP3/M4A)</label>
        <input type="file" id="audioFile" accept="audio/*">
        <button class="action" id="btnAudio" onclick="analyseAudio()">Analyse voice note</button>
      </div>

      <div class="toggles">
        <label class="toggle"><input type="checkbox" id="calibrateToggle">
          Use calibrated scoring (softens the over-confident classifier and folds in
          link &amp; benign-context evidence)</label>
        <label class="toggle"><input type="checkbox" id="explainToggle">
          Explain which words influenced the result (slower — runs the classifier
          once per word)</label>
      </div>

      <div class="errorbox" id="errorBox" role="alert"></div>

      <div class="result simple-mode" id="result" role="region" aria-label="Analysis result" aria-live="polite">
        <div class="viewswitch" role="group" aria-label="Level of detail">
          <button type="button" id="viewSimple" aria-pressed="true" onclick="setView('simple')">Simple view</button>
          <button type="button" id="viewDetailed" aria-pressed="false" onclick="setView('detailed')">Show details</button>
        </div>
        <div class="bottomline" id="bottomline"></div>
        <div class="card">
          <h3>Risk Level</h3>
          <span id="badge" class="badge"></span>
          <div class="meter" id="meter" role="progressbar" aria-label="Overall risk score"
               aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
            <div id="meterFill" class="meter-fill"></div>
          </div>
          <p id="summary" style="margin-top: 0.75rem;"></p>
          <p class="meta detail" id="meta"></p>
          <p class="source-note" id="sourceNote"></p>
        </div>

        <!-- "Why?" card.
             Shows the top 2-3 reasons in plain words, built ONLY from evidence
             the pipeline already returned (rule indicators, link findings,
             benign-context signals, classifier result). Visible in both views
             so the Simple view explains itself, not just a verdict. -->
        <div class="card" id="whyCard">
          <h3 id="whyTitle">Why TrustLens thinks this</h3>
          <ul class="why-list" id="whyList"></ul>
        </div>
        <div class="card detail" id="indicatorsCard">
          <h3>Suspicious Indicators</h3>
          <div id="indicators"></div>
        </div>
        <div class="card detail" id="attributionCard" style="display:none;">
          <h3>Influential Words</h3>
          <p class="attn-legend">Which words moved the phishing classifier.
            <span class="attn-swatch" style="background:#c0392b;"></span>raised risk
            <span class="attn-swatch" style="background:#2e7d32;"></span>lowered risk</p>
          <div id="attributionContent"></div>
        </div>
        <div class="card detail" id="linkCard" style="display:none;">
          <h3>Link Analysis</h3>
          <div id="linkContent"></div>
        </div>
        <div class="card detail" id="contextCard" style="display:none;">
          <h3>Legitimate-context Signals</h3>
          <div id="contextContent"></div>
        </div>
        <div class="card detail" id="reasoningCard">
          <h3>Reasoning</h3>
          <p id="reasoning"></p>
          <div class="emotion-box" id="emotionBox" style="display:none;"></div>
        </div>
        <div class="card">
          <h3>Recommended Action</h3>
          <div class="action-box" id="action"></div>
          <p class="disclaimer" id="disclaimer"></p>
        </div>

        <!-- "What should I do next?" with three choices.
             Progressive disclosure: the user only sees the steps for the
             option they pick. Static guidance; does not change any result. -->
        <div class="card" id="nextStepsCard">
          <h3 id="nextTitle">What should I do next?</h3>
          <div class="choices" role="group" aria-label="What should I do next?">
            <button type="button" class="secondary" data-choice="verify" aria-pressed="false"
                    aria-controls="nextPanel" onclick="chooseNext('verify')">Verify safely</button>
            <button type="button" class="secondary" data-choice="clicked" aria-pressed="false"
                    aria-controls="nextPanel" onclick="chooseNext('clicked')">I already clicked</button>
            <button type="button" class="secondary" data-choice="report" aria-pressed="false"
                    aria-controls="nextPanel" onclick="chooseNext('report')">Report / block it</button>
          </div>
          <div class="next-panel" id="nextPanel" aria-live="polite"></div>
          <p class="next-hint" id="nextHint">Choose an option to see short, practical steps.</p>
        </div>

        <!-- "Check another message".
             Clears this tab's input and result so the user can start again. -->
        <button type="button" class="secondary" id="btnAnother" onclick="checkAnother()">Check another message</button>

      </div>
    </main>
  </div>

  <script>
    const METER_COLOURS = { 'Safe':'#256a2a','Caution':'#9a6b00','Suspicious':'#b0500a','High Risk':'#a5201d' };
    const LEVEL_GLYPH   = { 'Safe':'\\u2713','Caution':'!','Suspicious':'\\u25B2','High Risk':'\\u2715' };
    // Plain-language at-a-glance verdict shown large at the top of the result.
    const BOTTOM_LINE = {
      'Safe':       'This looks safe. Stay cautious with unexpected requests.',
      'Caution':    'Be careful with this message. Verify it before you act.',
      'Suspicious': 'This is probably a scam. Do not click links or reply.',
      'High Risk':  'This looks like a scam. Do not click anything, reply, or call.'
    };

    // =================================================================
    // Interface helpers (presentation only)
    // The data and functions in this block only change what the page SHOWS.
    // It never changes the models, scores, thresholds or the API.
    // =================================================================

    // (1) "What should I do next?" -- short steps for each choice.
    //     General advice that stays correct across apps and devices
    //     (platform-specific steps change too often to keep accurate).
    const NEXT_ACTIONS = {
      verify: {
        title: 'Verify safely',
        steps: [
          'Do not use any link, phone number or QR code in the message.',
          'Open the organisation app yourself, or type its official website address.',
          'Or call the number on the back of your card or on its official website, and ask if they sent it.'
        ]
      },
      clicked: {
        title: 'If you already clicked or replied',
        steps: [
          'Close the page and do not enter anything else.',
          'If you typed a password, change it now through the official app or website, and turn on two-factor authentication.',
          'If you shared card or bank details, or a one-time code, call your bank immediately using its official number.',
          'Watch your accounts for payments you do not recognise.'
        ]
      },
      report: {
        title: 'Report or block it',
        steps: [
          'Use the Report or Block option in the app where you received it (SMS, WhatsApp, Telegram or email).',
          'In Singapore, you can also check or report it through ScamShield (helpline 1799).',
          'Delete the message once you have reported it.'
        ]
      }
    };

    // (2) Plain-language versions of the evidence names the pipeline returns.
    //     Keys match rule indicators (indicators.py) and link findings
    //     (link_analysis.py). Two keys can share a sentence; duplicates are
    //     removed before display.
    const PLAIN_REASONS = {
      urgency:              'It pressures you to act quickly.',
      credential_request:   'It asks for your password or login details.',
      suspicious_url_ip:    'Its link goes to a number (IP) address instead of a normal website name.',
      ip_literal:           'Its link goes to a number (IP) address instead of a normal website name.',
      payment_pressure:     'It mentions a payment problem or refund to get you to act.',
      prize_lure:           'It offers a prize or reward you did not expect.',
      impersonation_brand:  'It uses the name of a well-known company or organisation.',
      brand_url_mismatch:   'It names a company but links to a different website.',
      account_threat:       'It threatens to lock, suspend or close your account.',
      url_shortener:        'Its link is shortened, which hides where it really goes.',
      deceptive_subdomain:  'Its link puts a trusted name in front of a different website.',
      brand_not_registrable:'Its link uses a company name but is not that company website.',
      punycode:             'Its link uses look-alike letters to imitate a real website.',
      suspicious_tld:       'Its web address ends in a type often used by scammers.',
      userinfo_obfuscation: 'Its link hides the real destination after an @ sign.',
      non_standard_port:    'Its link uses an unusual technical setting that real companies rarely use.',
      hex_or_encoded:       'Its link is disguised with encoded characters.',
      excessive_subdomains: 'Its web address is unusually long and complicated.'
    };
    // Order in which rule indicators are shown (most important first).
    const INDICATOR_PRIORITY = ['credential_request', 'brand_url_mismatch', 'account_threat',
      'payment_pressure', 'prize_lure', 'urgency', 'suspicious_url_ip', 'impersonation_brand'];

    // Plain-language versions of the benign-context signals (context_signals.py).
    const BENIGN_REASONS = {
      order_reference:     'It quotes a specific order or reference number.',
      tracking_number:     'It includes a real-looking tracking number.',
      unsubscribe_footer:  'It has an unsubscribe footer, like genuine marketing email.',
      no_action_required:  'It does not ask you to do anything.',
      appointment_receipt: 'It uses routine delivery, booking or receipt wording.',
      known_good_domain:   'Its link goes to the company official website.'
    };
    const MAX_REASONS = 3;   // keep the Simple view short

    // (3) "Try an example" messages. Written for the demo; NOT taken from
    //     the 25-message evaluation set, so they do not affect evaluation.
    const EXAMPLES = {
      scam: 'Your PayPal account has been locked. Verify your password immediately at ' +
            'http://192.168.1.45/login or your account will be suspended.',
      safe: 'Hi, just checking you got home okay last night. Are you free for dinner on Friday?'
    };
    // ======================== end of interface data ========================

    let currentView = 'simple';   // default to the simpler, plain-language view
    function setView(mode) {
      currentView = mode;
      const result = document.getElementById('result');
      result.classList.toggle('simple-mode', mode === 'simple');
      document.getElementById('viewSimple').setAttribute('aria-pressed', mode === 'simple' ? 'true' : 'false');
      document.getElementById('viewDetailed').setAttribute('aria-pressed', mode === 'detailed' ? 'true' : 'false');
    }

    const TABS = ['text','image','audio'];

    // Per-tab results (presentation only).
    // Each input tab keeps its own last result, so switching tabs never shows
    // another tab's analysis. Models, scoring and the API are unchanged.
    let currentTab = 'text';
    const lastResult = { text: null, image: null, audio: null };

    function showTabResult(tab) {
      clearError();
      if (lastResult[tab]) { renderResult(lastResult[tab]); }
      else { document.getElementById('result').style.display = 'none'; }
    }

    function switchTab(el) {
      document.querySelectorAll('[role="tab"]').forEach(t => {
        const on = t === el;
        t.setAttribute('aria-selected', on ? 'true' : 'false');
        t.tabIndex = on ? 0 : -1;
      });
      TABS.forEach(name => {
        document.getElementById('panel-' + name).hidden = (name !== el.dataset.panel);
      });
      currentTab = el.dataset.panel;   // remember the active tab
      showTabResult(currentTab);       // show only this tab's result
    }

    // Keyboard support for the tablist (Left/Right/Home/End), per ARIA practices.
    document.querySelectorAll('[role="tab"]').forEach((tab, i, all) => {
      tab.addEventListener('keydown', e => {
        let idx = null;
        if (e.key === 'ArrowRight') idx = (i + 1) % all.length;
        else if (e.key === 'ArrowLeft') idx = (i - 1 + all.length) % all.length;
        else if (e.key === 'Home') idx = 0;
        else if (e.key === 'End') idx = all.length - 1;
        if (idx !== null) { e.preventDefault(); all[idx].focus(); switchTab(all[idx]); }
      });
    });

    function busy(btn, on, label) {
      btn.disabled = on;
      btn.setAttribute('aria-busy', on ? 'true' : 'false');
      btn.textContent = on ? 'Analysing...' : label;
    }
    function calib()   { return document.getElementById('calibrateToggle').checked; }
    function explain() { return document.getElementById('explainToggle').checked; }
    function showError(msg) {
      const box = document.getElementById('errorBox');
      box.textContent = msg; box.style.display = 'block';
    }
    function clearError() { document.getElementById('errorBox').style.display = 'none'; }

    async function analyseText() {
      clearError();
      const text = document.getElementById('input').value.trim();
      if (!text) { showError('Please enter a message to analyse.'); return; }
      const btn = document.getElementById('btnText'); busy(btn, true);
      try {
        const res = await fetch('/api/analyse', { method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, calibrate: calib(), explain: explain() }) });
        handle(await res.json(), 'text');
      } catch (e) { showError('Error: ' + e.message); } finally { busy(btn, false, 'Analyse text'); }
    }
    async function analyseImage() {
      clearError();
      const f = document.getElementById('imageFile').files[0];
      if (!f) { showError('Please choose an image first.'); return; }
      const btn = document.getElementById('btnImage'); busy(btn, true);
      const fd = new FormData(); fd.append('image', f);
      try { const res = await fetch('/api/analyse-image?calibrate=' + calib() + '&explain=' + explain(),
              { method: 'POST', body: fd });
        handle(await res.json(), 'image'); } catch (e) { showError('Error: ' + e.message); }
      finally { busy(btn, false, 'Analyse screenshot'); }
    }
    async function analyseAudio() {
      clearError();
      const f = document.getElementById('audioFile').files[0];
      if (!f) { showError('Please choose an audio file first.'); return; }
      const btn = document.getElementById('btnAudio'); busy(btn, true);
      const fd = new FormData(); fd.append('audio', f);
      try { const res = await fetch('/api/analyse-audio?calibrate=' + calib() + '&explain=' + explain(),
              { method: 'POST', body: fd });
        handle(await res.json(), 'audio'); } catch (e) { showError('Error: ' + e.message); }
      finally { busy(btn, false, 'Analyse voice note'); }
    }

    // ================= Interface functions (presentation only) =================

    // Fill the Text tab with an example and analyse it via the normal route.
    function tryExample(kind) {
      const box = document.getElementById('input');
      box.value = EXAMPLES[kind] || '';
      analyseText();
    }

    // Turn an evidence label back into its key, e.g. "Brand Url Mismatch" -> "brand_url_mismatch".
    function toKey(name) { return String(name || '').trim().toLowerCase().split(' ').join('_'); }

    // Build the plain-language "Why?" list from evidence the API already returned.
    function buildWhy(data) {
      const level = data.assessment.level;
      const reasons = [];
      const add = txt => { if (txt && reasons.indexOf(txt) === -1) reasons.push(txt); };

      if (level === 'Safe') {
        // Safe: explain what looked legitimate, or that nothing risky was found.
        const bc = data.explanation.benign_context;
        if (bc && bc.signals) bc.signals.forEach(sig => add(BENIGN_REASONS[sig.name]));
        if (!reasons.length) {
          add('No common scam warning signs were found.');
          add('The AI model did not find scam-like wording.');
        }
      } else {
        // Risky: strongest link findings first (sorted by severity), then rule indicators.
        const la = data.explanation.link_analysis;
        if (la && la.findings) {
          la.findings.slice().sort((a, b) => (b.severity || 0) - (a.severity || 0))
            .forEach(f => add(PLAIN_REASONS[f.name]));
        }
        const keys = (data.explanation.indicators || []).map(i => toKey(i.name));
        INDICATOR_PRIORITY.forEach(k => { if (keys.indexOf(k) !== -1) add(PLAIN_REASONS[k]); });
        keys.forEach(k => add(PLAIN_REASONS[k]));   // any indicator not in the priority list
        if (data.explanation.emotion_note) add('It uses emotional pressure to rush you.');
        // No explicit evidence: the result rests on the classifier alone. Say so honestly.
        if (!reasons.length) add('Its wording closely resembles known scam messages (AI model).');
      }

      document.getElementById('whyTitle').textContent =
        level === 'Safe' ? 'Why this looks okay' : 'Why this looks risky';
      const list = document.getElementById('whyList');
      list.innerHTML = '';
      reasons.slice(0, MAX_REASONS).forEach(r => {
        const li = document.createElement('li'); li.textContent = r; list.appendChild(li);
      });
    }

    // Reset the "What should I do next?" card to its collapsed state.
    function resetNext(level) {
      document.getElementById('nextTitle').textContent =
        level === 'Safe' ? 'Still unsure? What to do next' : 'What should I do next?';
      document.querySelectorAll('.choices button').forEach(b => b.setAttribute('aria-pressed', 'false'));
      document.getElementById('nextPanel').innerHTML = '';
      document.getElementById('nextHint').style.display = 'block';
    }

    // Show the steps for the chosen option (progressive disclosure).
    function chooseNext(choice) {
      const info = NEXT_ACTIONS[choice];
      if (!info) return;
      document.querySelectorAll('.choices button').forEach(b =>
        b.setAttribute('aria-pressed', b.dataset.choice === choice ? 'true' : 'false'));
      const panel = document.getElementById('nextPanel');
      panel.innerHTML = '';
      const h = document.createElement('strong'); h.textContent = info.title; panel.appendChild(h);
      const ol = document.createElement('ol');
      info.steps.forEach(step => { const li = document.createElement('li'); li.textContent = step; ol.appendChild(li); });
      panel.appendChild(ol);
      document.getElementById('nextHint').style.display = 'none';
    }

    // Clear the current tab input and result so the user can start again.
    function checkAnother() {
      clearError();
      if (currentTab === 'text') document.getElementById('input').value = '';
      if (currentTab === 'image') document.getElementById('imageFile').value = '';
      if (currentTab === 'audio') document.getElementById('audioFile').value = '';
      lastResult[currentTab] = null;
      document.getElementById('result').style.display = 'none';
      const focusId = { text: 'input', image: 'imageFile', audio: 'audioFile' }[currentTab];
      const el = document.getElementById(focusId);
      window.scrollTo({ top: 0, behavior: 'smooth' });
      if (el) el.focus();
    }
    // ============================ end of interface functions ============================

    function handle(data, tab) {
      tab = tab || currentTab;
      if (data.error && !data.assessment) {
        lastResult[tab] = null;                        // clear this tab's stored result
        if (tab === currentTab) {
          document.getElementById('result').style.display = 'none';
          showError(data.error);
        }
        return;
      }
      lastResult[tab] = data;                          // store per tab
      if (tab === currentTab) renderResult(data);      // only show on its own tab
    }
    function renderResult(data) {
      clearError();
      document.getElementById('result').style.display = 'block';
      const level = data.assessment.level;
      const badge = document.getElementById('badge');
      const glyph = LEVEL_GLYPH[level] || '';
      badge.className = 'badge badge-' + level.replace(' ', '');
      badge.innerHTML = '<span class="glyph" aria-hidden="true">' + glyph + '</span>' +
                        '<span>' + level + '</span>';
      badge.setAttribute('aria-label', 'Risk level: ' + level);

      // Big plain-language bottom line (shows in both views; the only rich text in Simple view)
      const bl = document.getElementById('bottomline');
      bl.className = 'bottomline bl-' + level.replace(' ', '');
      bl.innerHTML = '<span class="bl-glyph" aria-hidden="true">' + glyph + '</span>' +
                     '<span>' + (BOTTOM_LINE[level] || '') + '</span>';

      const pct = Math.round((data.assessment.score || 0) * 100);
      const fill = document.getElementById('meterFill');
      fill.style.width = pct + '%';
      fill.style.background = METER_COLOURS[level] || '#666';
      document.getElementById('meter').setAttribute('aria-valuenow', pct);

      document.getElementById('summary').textContent = data.explanation.summary;
      let meta = `Score: ${data.assessment.score} | Classifier: ${data.assessment.classifier_prob} | Indicators: ${data.assessment.indicator_count}`;
      if (data.assessment.emotion_pressure !== undefined) meta += ` | Emotion: ${data.assessment.emotion_pressure}`;
      if (data.assessment.calibrated) meta += ` | calibrated (T=${data.assessment.temperature})`;
      document.getElementById('meta').textContent = meta;

      const src = data.source;
      const note = document.getElementById('sourceNote');
      if (src === 'image') note.textContent = 'Text extracted from your screenshot via OCR, then analysed.';
      else if (src === 'audio') note.textContent = 'Speech transcribed from your voice note, then analysed.';
      else note.textContent = '';

      const indsDiv = document.getElementById('indicators');
      const indsCard = document.getElementById('indicatorsCard');
      indsDiv.innerHTML = '';
      if (!data.explanation.indicators.length) { indsCard.style.display = 'none'; }
      else { indsCard.style.display = 'block';
        data.explanation.indicators.forEach(ind => {
          const tag = document.createElement('span'); tag.className = 'indicator'; tag.textContent = ind.name;
          indsDiv.appendChild(tag);
          const exp = document.createElement('p'); exp.className = 'indicator-explain';
          exp.textContent = ind.reason; indsDiv.appendChild(exp); }); }

      // Influential-words card (opt-in occlusion attribution)
      const attnCard = document.getElementById('attributionCard');
      const attnContent = document.getElementById('attributionContent');
      attnContent.innerHTML = '';
      const attn = data.attribution;
      if (attn && attn.available && attn.tokens && attn.tokens.length) {
        attnCard.style.display = 'block';
        attn.tokens.forEach(t => {
          const row = document.createElement('div'); row.className = 'attn-row';
          const word = document.createElement('span');
          word.className = 'attn-word ' + t.direction; word.textContent = t.token;
          const bar = document.createElement('span'); bar.className = 'attn-bar';
          const barFill = document.createElement('span');
          barFill.className = 'attn-bar-fill ' + t.direction;
          barFill.style.width = Math.round((t.weight || 0) * 100) + '%';
          bar.appendChild(barFill);
          const dir = document.createElement('span'); dir.className = 'attn-dir';
          dir.textContent = t.direction === 'phishing' ? 'raised risk' : 'lowered risk';
          row.appendChild(word); row.appendChild(bar); row.appendChild(dir);
          attnContent.appendChild(row);
        });
      } else if (attn && !attn.available) {
        attnCard.style.display = 'block';
        const p = document.createElement('p'); p.className = 'indicator-explain';
        p.textContent = attn.note || 'Word explanation unavailable.';
        attnContent.appendChild(p);
      } else if (attn) {
        // Classifier ran but no single word moved it (typical when the model
        // is saturated at ~0 or ~1). Show the note instead of hiding the card.
        attnCard.style.display = 'block';
        const p = document.createElement('p'); p.className = 'indicator-explain';
        p.textContent = attn.note || 'No single word changed the classifier decision appreciably.';
        attnContent.appendChild(p);
      } else { attnCard.style.display = 'none'; }

      // Link Analysis card (advanced URL forensics)
      const linkCard = document.getElementById('linkCard');
      const linkContent = document.getElementById('linkContent');
      const la = data.explanation.link_analysis;
      linkContent.innerHTML = '';
      if (la && la.findings && la.findings.length) {
        linkCard.style.display = 'block';
        la.findings.forEach(f => {
          const tag = document.createElement('span'); tag.className = 'link-finding'; tag.textContent = f.name;
          linkContent.appendChild(tag);
          const exp = document.createElement('p'); exp.className = 'finding-explain';
          exp.textContent = f.reason; linkContent.appendChild(exp); });
      } else { linkCard.style.display = 'none'; }

      // Benign-context card
      const ctxCard = document.getElementById('contextCard');
      const ctxContent = document.getElementById('contextContent');
      const bc = data.explanation.benign_context;
      ctxContent.innerHTML = '';
      if (bc && bc.signals && bc.signals.length) {
        ctxCard.style.display = 'block';
        bc.signals.forEach(s => {
          const tag = document.createElement('span'); tag.className = 'ctx-finding'; tag.textContent = s.name;
          ctxContent.appendChild(tag);
          const exp = document.createElement('p'); exp.className = 'finding-explain';
          exp.textContent = s.reason; ctxContent.appendChild(exp); });
      } else { ctxCard.style.display = 'none'; }

      document.getElementById('reasoning').textContent = data.explanation.reasoning;
      const emoBox = document.getElementById('emotionBox');
      if (data.explanation.emotion_note) { emoBox.style.display = 'block'; emoBox.textContent = data.explanation.emotion_note; }
      else { emoBox.style.display = 'none'; }
      document.getElementById('action').textContent = data.explanation.safe_action;

      // Fill the plain-language "Why?" card and reset the
      // "What should I do next?" choices for this result. Display only.
      buildWhy(data);
      resetNext(level);

      document.getElementById('disclaimer').textContent = data.explanation.disclaimer;

      setView(currentView);   // keep the user's chosen Simple/Detailed view
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

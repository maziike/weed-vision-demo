import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


st.set_page_config(
    page_title="Blueberry Weed Detection · AI Vision Console",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.html(
    """
    <style>
    :root {
        --bg: #0d1117;
        --panel: #161b22;
        --panel-soft: #0d1117;
        --border: #30363d;
        --text: #e6edf3;
        --muted: #8b949e;
        --green: #3fb950;
        --blue: #58a6ff;
        --red: #f85149;
    }

    .stApp { background: var(--bg); color: var(--text); }
    html, body, [class*="st-"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    code, pre, .mono, [data-testid="stMetricValue"] {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    }
    .block-container { max-width: 1440px; padding: 1.8rem 2.5rem 4rem; }
    #MainMenu, footer, header { visibility: hidden; }

    .repo-bar {
        display: flex; align-items: center; justify-content: space-between;
        gap: 16px; padding-bottom: 18px; margin-bottom: 26px;
        border-bottom: 1px solid var(--border);
    }
    .top-divider { height: 1px; background: var(--border); margin: 18px 0 36px; }
    .repo-name { font: 600 14px ui-monospace, monospace; color: var(--blue); }
    .repo-name span { color: var(--muted); font-weight: 400; }
    .repo-meta { display: flex; gap: 8px; flex-wrap: wrap; }
    .badge {
        border: 1px solid var(--border); border-radius: 999px; padding: 5px 10px;
        color: var(--muted); background: var(--panel); font: 12px ui-monospace, monospace;
    }
    .badge.online { color: #7ee787; }
    .badge.online::before { content: ""; display: inline-block; width: 7px; height: 7px;
        border-radius: 50%; background: var(--green); margin-right: 7px; }

    .hero { margin: 10px 0 28px; }
    .eyebrow { color: var(--green); font: 600 12px ui-monospace, monospace;
        letter-spacing: .08em; text-transform: uppercase; }
    .hero h1 { margin: 8px 0 8px; color: var(--text); font-size: clamp(30px, 4vw, 48px);
        line-height: 1.08; letter-spacing: -.035em; }
    .hero p { max-width: 720px; margin: 0; color: var(--muted); font-size: 16px; line-height: 1.6; }

    .section-label { margin: 12px 0 10px; color: var(--muted);
        font: 600 12px ui-monospace, monospace; letter-spacing: .06em; text-transform: uppercase; }
    .section-label b { color: var(--text); }

    [data-testid="stFileUploader"] {
        border: 1px dashed var(--border); border-radius: 8px; background: var(--panel);
        padding: 6px;
    }
    [data-testid="stFileUploader"] section { background: transparent; }
    [data-testid="stFileUploader"] button {
        background: #21262d; color: var(--text); border: 1px solid var(--border);
    }
    .stButton > button {
        width: 100%; min-height: 44px; border-radius: 6px; font-weight: 600;
        background: #238636; color: white; border: 1px solid rgba(240,246,252,.1);
    }
    .stButton > button:hover { background: #2ea043; border-color: #3fb950; color: white; }
    [data-testid="stImage"] img { border: 1px solid var(--border); border-radius: 8px; }

    [data-testid="stMetric"] {
        background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--text); font-size: 23px; }

    .code-window { border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
        background: #0d1117; margin-top: 2px; }
    .code-head { display: flex; align-items: center; justify-content: space-between;
        padding: 10px 14px; border-bottom: 1px solid var(--border); background: var(--panel);
        color: var(--muted); font: 12px ui-monospace, monospace; }
    .code-head strong { color: var(--text); font-weight: 600; }
    .code-body { min-height: 330px; max-height: 470px; overflow: auto; padding: 16px 18px;
        white-space: pre-wrap; color: #c9d1d9; font: 12.5px/1.75 ui-monospace, monospace; }
    .ln { display: inline-block; width: 26px; color: #484f58; user-select: none; }
    .log-time { color: #6e7681; } .log-info { color: var(--blue); }
    .log-ok { color: #7ee787; } .log-target { color: #ffa198; }
    .code-comment { color: #8b949e; } .code-key { color: #ff7b72; }
    .code-string { color: #a5d6ff; }

    .pipeline { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px;
        background: var(--border); border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
        margin-top: 10px; }
    .pipe-step { background: var(--panel); padding: 14px 12px; min-height: 72px; }
    .pipe-step small { display:block; color:#6e7681; font:11px ui-monospace,monospace; }
    .pipe-step b { display:block; color:var(--text); font:12px ui-monospace,monospace; margin-top:5px; }
    .target-row { border-bottom: 1px solid #21262d; padding: 8px 0; color: var(--muted);
        font: 12px ui-monospace, monospace; }
    .target-row b { color: var(--text); }
    @media (max-width: 800px) {
        .block-container { padding: 1.2rem; }
        .repo-bar { align-items:flex-start; flex-direction:column; }
        .pipeline { grid-template-columns: 1fr; }
    }
    </style>
    """
)


MODEL_ID = "weed-detection-bounding-boxes/2"
API_URL = "https://serverless.roboflow.com"


def get_api_key():
    """Load the Roboflow credential without storing it in source control."""
    try:
        api_key = st.secrets.get("ROBOFLOW_API_KEY")
    except Exception:
        api_key = None

    api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not configured. Add it in Streamlit Secrets."
        )
    return api_key


def run_inference(image, api_key):
    """Call Roboflow's hosted endpoint without importing the OpenCV-heavy SDK."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("ascii")
    response = requests.post(
        f"{API_URL}/{MODEL_ID}",
        params={"api_key": api_key},
        data=encoded_image,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def annotate_image(image, predictions):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    targets = []

    for index, pred in enumerate(predictions, start=1):
        x, y = pred["x"], pred["y"]
        width, height = pred["width"], pred["height"]
        confidence = pred["confidence"]
        x1, y1 = int(x - width / 2), int(y - height / 2)
        x2, y2 = int(x + width / 2), int(y + height / 2)
        target_x, target_y = int(x), int(y + height / 2)

        draw.rectangle([x1, y1, x2, y2], outline="#3fb950", width=5)
        label = f" WEED {confidence * 100:.0f}% "
        label_box = draw.textbbox((x1, max(0, y1 - 18)), label, font=font)
        draw.rectangle(label_box, fill="#238636")
        draw.text((x1, max(0, y1 - 18)), label, fill="white", font=font)

        radius = 9
        draw.ellipse(
            [target_x - radius, target_y - radius, target_x + radius, target_y + radius],
            fill="#f85149",
        )
        draw.line((target_x - 16, target_y, target_x + 16, target_y), fill="white", width=2)
        draw.line((target_x, target_y - 16, target_x, target_y + 16), fill="white", width=2)
        draw.text((target_x + 13, target_y + 9), f"TARGET {index}", fill="#f85149", font=font)
        targets.append((target_x, target_y, confidence))

    return output, targets


def code_panel(filename, targets=None, status="ready", error=None):
    lines = [
        '<span class="code-comment"># realtime_detection.py</span>',
        '<span class="code-key">import</span> requests  <span class="code-comment"># Roboflow hosted inference</span>',
        '',
        f'<span class="log-time">[00:00.000]</span> <span class="log-info">INFO</span>  model.resolve(<span class="code-string">"{MODEL_ID}"</span>)',
    ]
    if filename:
        lines.append(f'<span class="log-time">[00:00.012]</span> <span class="log-ok">INPUT</span> image.open(<span class="code-string">"{filename}"</span>)')
    else:
        lines.append('<span class="log-time">[--:--.---]</span> <span class="code-comment">WAIT  image input required</span>')

    if status == "running":
        lines.append('<span class="log-time">[00:00.031]</span> <span class="log-info">RUN</span>   requests.post(model_endpoint, image)')
    elif error:
        lines.append(f'<span class="log-time">[00:00.031]</span> <span class="log-target">ERROR</span> {error}')
    elif targets is not None:
        lines.append(f'<span class="log-time">[00:01.284]</span> <span class="log-ok">DONE</span>  predictions.count = {len(targets)}')
        for i, (x, y, confidence) in enumerate(targets, start=1):
            lines.append(
                f'<span class="log-time">[{i:02d}]</span> <span class="log-target">TARGET</span> '
                f'pixel=({x:04d}, {y:04d})  confidence={confidence:.3f}'
            )
        lines.append('<span class="log-time">[00:01.291]</span> <span class="log-ok">READY</span> target_pipeline.publish()')
    else:
        lines.append('<span class="log-time">[--:--.---]</span> <span class="code-comment">IDLE  awaiting detection command</span>')

    numbered = "\n".join(f'<span class="ln">{i}</span>{line}' for i, line in enumerate(lines, 1))
    return f"""
    <div class="code-window">
      <div class="code-head"><strong>Detection Code</strong><span>main · live</span></div>
      <div class="code-body">{numbered}</div>
    </div>
    """


st.html(
    """
    <div class="top-divider"></div>
    <div class="hero">
      <h1>Blueberry Weed Detection,<br>engineered for action.</h1>
      <p>Identify weeds in blueberry crops, calculate extraction targets and hand precise image coordinates to the robotic removal pipeline.</p>
    </div>
    """
)

if "detection" not in st.session_state:
    st.session_state.detection = None
if "detection_error" not in st.session_state:
    st.session_state.detection_error = None

st.html('<div class="section-label">01 / <b>AI Detection</b></div>')
uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    help="JPG or PNG crop image",
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    file_signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("file_signature") != file_signature:
        st.session_state.file_signature = file_signature
        st.session_state.detection = None
        st.session_state.detection_error = None

    image_col, code_col = st.columns([1.12, 0.88], gap="large")
    with image_col:
        st.html('<div class="section-label"><b>Camera input</b> · RGB frame</div>')
        detection = st.session_state.detection
        st.image(
            detection["output"] if detection else image,
            use_container_width=True,
            caption="Annotated inference result" if detection else "Source image · ready for inference",
        )
        detect_clicked = st.button("Run detection", type="primary", use_container_width=True)

    with code_col:
        st.html('<div class="section-label"><b>Runtime</b> · inference stream</div>')
        code_slot = st.empty()
        current_targets = st.session_state.detection["targets"] if st.session_state.detection else None
        code_slot.html(code_panel(uploaded_file.name, current_targets, error=st.session_state.detection_error))

    if detect_clicked:
        code_slot.html(code_panel(uploaded_file.name, status="running"))
        st.session_state.detection_error = None
        try:
            with st.spinner("Running inference…"):
                result = run_inference(image, get_api_key())
                predictions = result.get("predictions", [])
                output, targets = annotate_image(image, predictions)
                st.session_state.detection = {
                    "output": output,
                    "targets": targets,
                    "predictions": predictions,
                }
        except Exception as exc:
            st.session_state.detection = None
            st.session_state.detection_error = str(exc)
        st.rerun()

    detection = st.session_state.detection
    if st.session_state.detection_error:
        st.error(f"Detection failed: {st.session_state.detection_error}")
    elif detection:
        targets = detection["targets"]
        avg_conf = sum(item[2] for item in targets) / len(targets) if targets else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("Objects", len(targets))
        m2.metric("Average confidence", f"{avg_conf * 100:.1f}%")
        m3.metric("System", "TARGET READY" if targets else "CLEAR")

        st.html('<div class="section-label">02 / <b>Target coordinates</b></div>')
        if targets:
            target_html = "".join(
                f'<div class="target-row"><b>target_{i:02d}</b> &nbsp; pixel=({x}, {y}) &nbsp; confidence={confidence * 100:.1f}%</div>'
                for i, (x, y, confidence) in enumerate(targets, start=1)
            )
            st.html(target_html)
        else:
            st.info("No weeds detected in this frame.")
else:
    st.html(code_panel(None))
    st.caption("Upload a crop image to initialize the detection pipeline.")

st.html(
    """
    <div class="section-label">03 / <b>Target pipeline</b></div>
    <div class="pipeline">
      <div class="pipe-step"><small>01 · perceive</small><b>RGB DETECTION</b></div>
      <div class="pipe-step"><small>02 · locate</small><b>PIXEL TARGET</b></div>
      <div class="pipe-step"><small>03 · resolve</small><b>DEPTH XYZ</b></div>
      <div class="pipe-step"><small>04 · transform</small><b>ARM COORDINATE</b></div>
      <div class="pipe-step"><small>05 · execute</small><b>EXTRACTION</b></div>
    </div>
    """
)

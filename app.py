import base64
import os
from io import BytesIO

import requests
import streamlit as st
import streamlit.components.v1 as components
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
ROI_TOP_RATIO = 0.42


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


def code_panel(
    filename,
    targets=None,
    raw_count=None,
    outside_roi_count=None,
    status="ready",
    error=None,
):
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
        raw_count = len(targets) if raw_count is None else raw_count
        outside_roi_count = 0 if outside_roi_count is None else outside_roi_count
        lines.extend(
            [
                f'<span class="log-time">[00:01.284]</span> <span class="log-info">RAW</span>   predictions.count = {raw_count}',
                f'<span class="log-time">[00:01.285]</span> <span class="log-target">FILTER</span> outside_roi = {outside_roi_count}',
                f'<span class="log-time">[00:01.286]</span> <span class="log-ok">VALID</span> targets.count = {len(targets)}',
            ]
        )
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


def render_robot_viewer(model_path):
    """Render a self-contained GLB model inside a Three.js iframe."""
    if not os.path.exists(model_path):
        st.warning("Robot model is unavailable. Add assets/robot_arm.glb to the project.")
        return

    with open(model_path, "rb") as model_file:
        model_data = base64.b64encode(model_file.read()).decode("ascii")

    viewer_html = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        * { box-sizing: border-box; }
        html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #0d1117; }
        #viewer { position: relative; width: 100%; height: 620px; border: 1px solid #30363d;
          border-radius: 8px; overflow: hidden; background: radial-gradient(circle at 50% 35%, #18212b 0, #0d1117 62%); }
        canvas { display: block; width: 100%; height: 100%; }
        .hud { position: absolute; z-index: 2; pointer-events: none; font-family: ui-monospace,
          SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        .title { top: 16px; left: 18px; color: #e6edf3; font-size: 13px; font-weight: 700; }
        .title small { display: block; color: #8b949e; font-size: 10px; font-weight: 400;
          letter-spacing: .08em; margin-top: 5px; }
        .status { top: 16px; right: 18px; padding: 6px 9px; border: 1px solid #30363d;
          border-radius: 999px; color: #7ee787; background: rgba(22,27,34,.88); font-size: 10px; }
        .status::before { content: ""; display: inline-block; width: 6px; height: 6px;
          border-radius: 50%; background: #3fb950; margin-right: 6px; box-shadow: 0 0 7px #3fb950; }
        .help { bottom: 15px; left: 18px; color: #8b949e; font-size: 10px;
          background: rgba(13,17,23,.78); padding: 7px 9px; border-radius: 5px; }
        .controls { right: 16px; bottom: 16px; width: 285px; padding: 13px 14px;
          border: 1px solid #30363d; border-radius: 7px; background: rgba(13,17,23,.92);
          color: #c9d1d9; font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; }
        .control-head { display: flex; justify-content: space-between; align-items: center;
          margin-bottom: 10px; color: #e6edf3; font-weight: 700; }
        .control-head button { pointer-events: auto; border: 1px solid #30363d; border-radius: 4px;
          padding: 4px 7px; background: #21262d; color: #c9d1d9; cursor: pointer;
          font: 9px ui-monospace, monospace; }
        .joint { display: grid; grid-template-columns: 65px 1fr 38px; gap: 7px;
          align-items: center; margin: 8px 0; }
        .joint label { color: #8b949e; }
        .joint output { color: #7ee787; text-align: right; }
        .joint input { width: 100%; accent-color: #3fb950; pointer-events: auto; cursor: pointer; }
        .action-row { display: grid; grid-template-columns: 1fr auto; gap: 7px; margin-top: 11px;
          padding-top: 11px; border-top: 1px solid #30363d; }
        .action-row button { pointer-events: auto; border-radius: 5px; padding: 8px 9px; cursor: pointer;
          font: 700 9px ui-monospace, monospace; letter-spacing: .04em; }
        #run-sequence { color: #fff; background: #238636; border: 1px solid #3fb950; }
        #run-sequence:hover:not(:disabled) { background: #2ea043; }
        #run-sequence:disabled, .controls input:disabled { cursor: not-allowed; opacity: .45; }
        .sequence-state { margin-top: 8px; min-height: 12px; color: #58a6ff; font-size: 9px; }
        .sim-badge { color: #d29922; }
        .loading { inset: 0; display: grid; place-items: center; color: #58a6ff; font-size: 11px;
          letter-spacing: .08em; background: #0d1117; transition: opacity .35s ease; }
        .loading.done { opacity: 0; }
        @media (max-width: 700px) {
          .controls { width: calc(100% - 32px); }
          .help { display: none; }
        }
      </style>
      <script type="importmap">
        {"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"}}
      </script>
    </head>
    <body>
      <div id="viewer">
        <div class="hud title">ROBOT_ARM / CAD_01<small>DIGITAL TWIN · GLTF 2.0</small></div>
        <div class="hud status" id="status">MODEL LOADING</div>
        <div class="hud help">DRAG · ROTATE &nbsp;&nbsp; SCROLL · ZOOM &nbsp;&nbsp; RIGHT DRAG · PAN</div>
        <div class="hud controls">
          <div class="control-head"><span>JOINT CONTROL</span><button id="reset-joints">RESET</button></div>
          <div class="joint"><label>BASE</label><input id="base" type="range" min="-90" max="90" value="0"><output>0°</output></div>
          <div class="joint"><label>SHOULDER</label><input id="shoulder" type="range" min="-55" max="75" value="0"><output>0°</output></div>
          <div class="joint"><label>ELBOW</label><input id="elbow" type="range" min="-95" max="95" value="0"><output>0°</output></div>
          <div class="joint"><label>WRIST</label><input id="wrist" type="range" min="-90" max="90" value="0"><output>0°</output></div>
          <div class="joint"><label>GRIPPER</label><input id="gripper" type="range" min="0" max="35" value="0"><output>0°</output></div>
          <div class="action-row"><button id="run-sequence" disabled>START AUTO EXTRACTION</button><span class="sim-badge">AUTO IK</span></div>
          <div class="sequence-state" id="sequence-state">SEARCHING FOR A REACHABLE TARGET...</div>
        </div>
        <div class="hud loading" id="loading">INITIALIZING GEOMETRY...</div>
      </div>
      <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

        const container = document.getElementById('viewer');
        const loading = document.getElementById('loading');
        const status = document.getElementById('status');
        const scene = new THREE.Scene();
        scene.fog = new THREE.Fog(0x0d1117, 10, 24);

        const camera = new THREE.PerspectiveCamera(38, container.clientWidth / container.clientHeight, 0.01, 100);
        camera.position.set(5.5, 3.8, 7.5);

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.15;
        container.prepend(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.target.set(0, 1.8, 0);
        controls.minDistance = 3;
        controls.maxDistance = 16;

        scene.add(new THREE.HemisphereLight(0xbad7ff, 0x17231c, 2.2));
        const key = new THREE.DirectionalLight(0xffffff, 3.2);
        key.position.set(5, 8, 5);
        scene.add(key);
        const rim = new THREE.DirectionalLight(0x58a6ff, 2.0);
        rim.position.set(-5, 4, -4);
        scene.add(rim);

        const grid = new THREE.GridHelper(14, 28, 0x238636, 0x26313c);
        grid.material.opacity = 0.45;
        grid.material.transparent = true;
        scene.add(grid);

        const joints = {};
        const gripper = {};
        const deg = THREE.MathUtils.degToRad;
        const sequenceState = document.getElementById('sequence-state');
        const runButton = document.getElementById('run-sequence');
        let gripPoint = null;
        let weedTarget = null;
        let selectedTargetPosition = null;
        let selectedSolution = null;
        let interactionPlane = null;
        let controlsLocked = false;
        const workspaceCenter = new THREE.Vector3();
        let sequenceGeneration = 0;
        const activeTweens = new Set();

        const normalizedName = (name) => name.toLowerCase().replace(/[^a-z0-9\u4e00-\u9fff]/g, '');
        function findPart(root, name) {
          const expected = normalizedName(name);
          let match = null;
          root.traverse((part) => {
            if (!match && normalizedName(part.name || '') === expected) match = part;
          });
          return match;
        }

        function jointPointBetween(meshA, meshB) {
          const boxA = new THREE.Box3().setFromObject(meshA);
          const boxB = new THREE.Box3().setFromObject(meshB);
          const overlap = boxA.clone().intersect(boxB);
          if (!overlap.isEmpty()) return overlap.getCenter(new THREE.Vector3());
          return boxA.getCenter(new THREE.Vector3())
            .add(boxB.getCenter(new THREE.Vector3())).multiplyScalar(0.5);
        }

        function buildJawPivot(assembly, name, partNames) {
          const pivot = new THREE.Group();
          pivot.name = name;
          const parts = partNames.map((partName) => findPart(assembly, partName)).filter(Boolean);
          if (!parts.length) {
            assembly.add(pivot);
            return pivot;
          }
          pivot.position.copy(parts[0].position);
          assembly.add(pivot);
          parts.forEach((part) => pivot.attach(part));
          return pivot;
        }

        function buildJointRig(robot) {
          const armRoot = findPart(robot, '机械臂');
          if (!armRoot) throw new Error('Mechanical arm root node not found');

          const waistMesh = findPart(armRoot, 'Waist-1');
          const arm01Mesh = findPart(armRoot, 'Arm 01-1');
          const arm02Mesh = findPart(armRoot, 'Arm 02 v3-1');
          const arm03Mesh = findPart(armRoot, 'Arm 03-1');
          const gripperAssembly = findPart(armRoot, 'Gripper Assembly');
          const servoShoulder = findPart(armRoot, 'Servo Motor MG996R-2');
          const servoElbow = findPart(armRoot, 'Servo Motor MG996R-7');
          const servoWrist1 = findPart(armRoot, 'Servo Motor Micro 9g-1');
          const servoWrist2 = findPart(armRoot, 'Servo Motor Micro 9g-2');
          if (![waistMesh, arm01Mesh, arm02Mesh, arm03Mesh, gripperAssembly].every(Boolean)) {
            throw new Error('Required CAD nodes are missing');
          }

          robot.updateMatrixWorld(true);
          const waistArm01 = jointPointBetween(waistMesh, arm01Mesh);
          const arm01Arm02 = jointPointBetween(arm01Mesh, arm02Mesh);
          const arm02Arm03 = jointPointBetween(arm02Mesh, arm03Mesh);
          const waistCenter = new THREE.Box3().setFromObject(waistMesh).getCenter(new THREE.Vector3());

          const armMount = new THREE.Group();
          armMount.name = 'RobotArmMount';
          robot.add(armMount);
          const base = new THREE.Group();
          const shoulder = new THREE.Group();
          const elbow = new THREE.Group();
          const wrist = new THREE.Group();
          base.name = 'joint_base';
          shoulder.name = 'joint_shoulder';
          elbow.name = 'joint_elbow';
          wrist.name = 'joint_wrist';
          base.position.set(waistCenter.x, 0, waistCenter.z);
          armMount.add(base);
          shoulder.position.copy(waistArm01.clone().sub(base.position));
          base.add(shoulder);
          const shoulderAbs = base.position.clone().add(shoulder.position);
          elbow.position.copy(arm01Arm02.clone().sub(shoulderAbs));
          shoulder.add(elbow);
          const elbowAbs = shoulderAbs.clone().add(elbow.position);
          wrist.position.copy(arm02Arm03.clone().sub(elbowAbs));
          elbow.add(wrist);

          base.attach(waistMesh);
          if (servoShoulder) base.attach(servoShoulder);
          shoulder.attach(arm01Mesh);
          elbow.attach(arm02Mesh);
          if (servoElbow) elbow.attach(servoElbow);
          wrist.attach(arm03Mesh);
          if (servoWrist1) wrist.attach(servoWrist1);
          if (servoWrist2) wrist.attach(servoWrist2);
          wrist.attach(gripperAssembly);

          joints.base = base;
          joints.shoulder = shoulder;
          joints.elbow = elbow;
          joints.wrist = wrist;
          gripper.left = buildJawPivot(gripperAssembly, 'GripperLeft', [
            'Gripper 1-2', 'gear1-1', 'grip link 1-1', 'grip link 1-4'
          ]);
          gripper.right = buildJawPivot(gripperAssembly, 'GripperRight', [
            'Gripper 1-1', 'gear2-1', 'grip link 1-2', 'grip link 1-3'
          ]);
          gripPoint = new THREE.Group();
          gripPoint.name = 'GripPoint';
          gripPoint.position.copy(gripper.left.position.clone().add(gripper.right.position).multiplyScalar(0.8));
          gripperAssembly.add(gripPoint);
        }

        function setJawAngle(angle) {
          if (gripper.left) gripper.left.rotation.y = angle;
          if (gripper.right) gripper.right.rotation.y = -angle;
        }

        function updateJoint(name, value) {
          const angle = deg(Number(value));
          if (name === 'base' && joints.base) joints.base.rotation.y = angle;
          if (name === 'shoulder' && joints.shoulder) joints.shoulder.rotation.x = angle;
          if (name === 'elbow' && joints.elbow) joints.elbow.rotation.y = angle;
          if (name === 'wrist' && joints.wrist) joints.wrist.rotation.x = angle;
          if (name === 'gripper') setJawAngle(angle);
        }

        function syncControls() {
          const values = {
            base: THREE.MathUtils.radToDeg(joints.base?.rotation.y || 0),
            shoulder: THREE.MathUtils.radToDeg(joints.shoulder?.rotation.x || 0),
            elbow: THREE.MathUtils.radToDeg(joints.elbow?.rotation.y || 0),
            wrist: THREE.MathUtils.radToDeg(joints.wrist?.rotation.x || 0),
            gripper: Math.abs(THREE.MathUtils.radToDeg(gripper.left?.rotation.y || 0)),
          };
          Object.entries(values).forEach(([name, value]) => {
            const input = document.getElementById(name);
            input.value = Math.round(value);
            input.nextElementSibling.value = `${Math.round(value)}°`;
          });
        }

        document.querySelectorAll('.joint input').forEach((input) => {
          input.addEventListener('input', () => {
            input.nextElementSibling.value = `${input.value}°`;
            updateJoint(input.id, input.value);
          });
        });

        function setControlsDisabled(disabled) {
          controlsLocked = disabled;
          document.querySelectorAll('.joint input').forEach((input) => { input.disabled = disabled; });
          runButton.disabled = disabled || !selectedTargetPosition || !selectedSolution;
        }

        function easeInOutCubic(t) {
          return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        }
        function tween(target, values, duration) {
          return new Promise((resolve) => {
            activeTweens.add({ target, values, duration, elapsed: 0,
              from: Object.fromEntries(Object.keys(values).map((key) => [key, target[key]])), resolve });
          });
        }
        function updateTweens(dt) {
          for (const item of [...activeTweens]) {
            item.elapsed += dt;
            const t = Math.min(1, item.elapsed / item.duration);
            const eased = easeInOutCubic(t);
            Object.keys(item.values).forEach((key) => {
              item.target[key] = item.from[key] + (item.values[key] - item.from[key]) * eased;
            });
            if (t >= 1) {
              activeTweens.delete(item);
              item.resolve();
            }
          }
        }
        const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

        function cancelTweens() {
          for (const item of activeTweens) item.resolve();
          activeTweens.clear();
        }

        function buildWeedTarget() {
          const target = new THREE.Group();
          const plant = new THREE.Group();
          const stem = new THREE.Mesh(
            new THREE.CylinderGeometry(0.035, 0.055, 0.55, 8),
            new THREE.MeshStandardMaterial({ color: 0x3fb950, roughness: 0.75 })
          );
          stem.position.y = 0.27;
          plant.add(stem);
          for (let i = 0; i < 6; i++) {
            const leaf = new THREE.Mesh(
              new THREE.ConeGeometry(0.11, 0.42, 5),
              new THREE.MeshStandardMaterial({ color: i % 2 ? 0x56d364 : 0x2ea043, side: THREE.DoubleSide })
            );
            const a = i / 6 * Math.PI * 2;
            leaf.position.set(Math.cos(a) * 0.05, 0.38, Math.sin(a) * 0.05);
            leaf.rotation.set(Math.sin(a) * 0.75, a, Math.cos(a) * 0.75);
            plant.add(leaf);
          }
          const ring = new THREE.Mesh(
            new THREE.RingGeometry(0.28, 0.34, 36),
            new THREE.MeshBasicMaterial({ color: 0xf85149, transparent: true, opacity: 0.9, side: THREE.DoubleSide })
          );
          ring.rotation.x = -Math.PI / 2;
          ring.position.y = 0.012;
          target.add(plant, ring);
          target.visible = false;
          scene.add(target);
          return { target, plant, ring, pulse: 0 };
        }

        function setupTargetPlacement() {
          // Visual guide only: every click still goes through the IK reachability check.
          const reachGuide = new THREE.Mesh(
            new THREE.RingGeometry(1.1, 4.1, 72),
            new THREE.MeshBasicMaterial({
              color: 0x3fb950,
              transparent: true,
              opacity: 0.16,
              side: THREE.DoubleSide,
              depthWrite: false
            })
          );
          reachGuide.rotation.x = -Math.PI / 2;
          reachGuide.position.set(workspaceCenter.x, 0.006, workspaceCenter.z);
          scene.add(reachGuide);

          interactionPlane = new THREE.Mesh(
            new THREE.PlaneGeometry(12, 12),
            new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, side: THREE.DoubleSide })
          );
          interactionPlane.rotation.x = -Math.PI / 2;
          interactionPlane.position.y = 0.008;
          scene.add(interactionPlane);

          const raycaster = new THREE.Raycaster();
          const pointer = new THREE.Vector2();
          renderer.domElement.addEventListener('pointerdown', (event) => {
            if (!event.shiftKey || controlsLocked) return;
            event.preventDefault();
            event.stopImmediatePropagation();
            const rect = renderer.domElement.getBoundingClientRect();
            pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            raycaster.setFromCamera(pointer, camera);
            const hit = raycaster.intersectObject(interactionPlane, false)[0];
            if (!hit) return;

            const radius = Math.hypot(
              hit.point.x - workspaceCenter.x,
              hit.point.z - workspaceCenter.z
            );
            if (radius < 1.1 || radius > 4.1) {
              status.textContent = 'OUTSIDE WORKSPACE';
              sequenceState.textContent = 'PLACE WEED INSIDE GREEN CANDIDATE AREA';
              return;
            }

            const candidate = hit.point.clone();
            candidate.y = 0;
            selectedTargetPosition = candidate;
            selectedSolution = null;
            weedTarget.target.attach(weedTarget.plant);
            weedTarget.plant.position.set(0, 0, 0);
            weedTarget.target.position.copy(candidate);
            weedTarget.target.visible = true;
            weedTarget.plant.visible = true;
            weedTarget.ring.visible = true;

            const grabCandidate = candidate.clone();
            grabCandidate.y = 0.42;
            const variables = jointVariables();
            const savedPose = variables.map((v) => v.joint.rotation[v.axis]);
            status.textContent = 'CHECKING REACH';
            sequenceState.textContent = 'IK VALIDATION IN PROGRESS';
            const candidateSolution = solveIK(grabCandidate);
            variables.forEach((v, index) => { v.joint.rotation[v.axis] = savedPose[index]; });
            scene.updateMatrixWorld(true);

            if (candidateSolution.error > 0.22) {
              selectedSolution = null;
              status.textContent = 'POINT UNREACHABLE';
              sequenceState.textContent = `WEED PLACED · IK ERROR ${candidateSolution.error.toFixed(2)}m · TRY ANOTHER POINT`;
              setControlsDisabled(false);
              return;
            }

            selectedSolution = candidateSolution;
            status.textContent = 'TARGET PLACED';
            sequenceState.textContent = `TARGET X ${selectedTargetPosition.x.toFixed(2)} / Z ${selectedTargetPosition.z.toFixed(2)}`;
            setControlsDisabled(false);
          }, true);
        }

        function prepareAutoTarget() {
          const variables = jointVariables();
          const savedPose = variables.map((v) => v.joint.rotation[v.axis]);
          let bestCandidate = null;

          // Search around the real CAD base and keep the target with the lowest IK error.
          const radii = [1.45, 1.9, 2.35, 2.8, 3.25, 3.7];
          const angles = [0, 45, 90, 135, 180, 225, 270, 315].map(deg);
          search:
          for (const radius of radii) {
            for (const angle of angles) {
              const position = new THREE.Vector3(
                workspaceCenter.x + Math.cos(angle) * radius,
                0,
                workspaceCenter.z + Math.sin(angle) * radius
              );
              const grabTarget = position.clone();
              grabTarget.y = 0.42;
              const solution = solveIK(grabTarget);
              if (!bestCandidate || solution.error < bestCandidate.solution.error) {
                bestCandidate = { position, solution };
              }
              if (solution.error < 0.1) break search;
            }
          }

          variables.forEach((v, index) => { v.joint.rotation[v.axis] = savedPose[index]; });
          scene.updateMatrixWorld(true);

          if (!bestCandidate || bestCandidate.solution.error > 0.22) {
            selectedTargetPosition = null;
            selectedSolution = null;
            weedTarget.target.visible = false;
            status.textContent = 'AUTO TARGET ERROR';
            sequenceState.textContent = 'NO SAFE IK TARGET FOUND';
            setControlsDisabled(false);
            return;
          }

          selectedTargetPosition = bestCandidate.position;
          selectedSolution = bestCandidate.solution;
          weedTarget.target.attach(weedTarget.plant);
          weedTarget.plant.position.set(0, 0, 0);
          weedTarget.target.position.copy(selectedTargetPosition);
          weedTarget.target.visible = true;
          weedTarget.plant.visible = true;
          weedTarget.ring.visible = true;
          status.textContent = 'AUTO TARGET READY';
          sequenceState.textContent = `IK LOCKED · ERROR ${selectedSolution.error.toFixed(2)}m · PRESS START`;
          setControlsDisabled(false);
        }

        function resetRig(clearTarget = true) {
          sequenceGeneration++;
          cancelTweens();
          if (joints.base) joints.base.rotation.set(0, 0, 0);
          if (joints.shoulder) joints.shoulder.rotation.set(0, 0, 0);
          if (joints.elbow) joints.elbow.rotation.set(0, 0, 0);
          if (joints.wrist) joints.wrist.rotation.set(0, 0, 0);
          setJawAngle(deg(12));
          if (weedTarget) {
            weedTarget.target.attach(weedTarget.plant);
            weedTarget.plant.position.set(0, 0, 0);
            if (clearTarget) {
              selectedTargetPosition = null;
              selectedSolution = null;
              weedTarget.plant.visible = false;
              weedTarget.target.visible = false;
            }
          }
          status.textContent = 'ARM READY';
          sequenceState.textContent = selectedTargetPosition
            ? 'TARGET READY · PRESS MOVE TO TARGET'
            : 'GREEN = CANDIDATE · SHIFT + CLICK TO IK CHECK';
          setControlsDisabled(false);
          syncControls();
        }

        function jointVariables() {
          return [
            { name: 'base', joint: joints.base, axis: 'y', min: deg(-90), max: deg(90) },
            { name: 'shoulder', joint: joints.shoulder, axis: 'x', min: deg(-55), max: deg(75) },
            { name: 'elbow', joint: joints.elbow, axis: 'y', min: deg(-105), max: deg(105) },
            { name: 'wrist', joint: joints.wrist, axis: 'x', min: deg(-90), max: deg(90) },
          ];
        }

        function solveIK(target) {
          const variables = jointVariables();
          const effector = new THREE.Vector3();
          const shifted = new THREE.Vector3();
          const epsilon = 0.004;
          const damping = 0.04;
          let best = { error: Infinity, pose: null };
          const seeds = [];
          for (const base of [-85, -45, 0, 45, 85]) {
            for (const shoulder of [-40, 10, 60]) {
              for (const elbow of [-85, 0, 85]) {
                seeds.push([base, shoulder, elbow, 0]);
              }
            }
          }

          for (const seed of seeds) {
            variables.forEach((v, index) => { v.joint.rotation[v.axis] = deg(seed[index]); });

            for (let iteration = 0; iteration < 85; iteration++) {
              scene.updateMatrixWorld(true);
              gripPoint.getWorldPosition(effector);
              const error = target.clone().sub(effector);
              if (error.length() < 0.08) break;

              const columns = variables.map((variable) => {
                const original = variable.joint.rotation[variable.axis];
                variable.joint.rotation[variable.axis] = original + epsilon;
                scene.updateMatrixWorld(true);
                gripPoint.getWorldPosition(shifted);
                variable.joint.rotation[variable.axis] = original;
                return shifted.clone().sub(effector).divideScalar(epsilon);
              });

              let xx = damping, xy = 0, xz = 0, yy = damping, yz = 0, zz = damping;
              columns.forEach((c) => {
                xx += c.x * c.x; xy += c.x * c.y; xz += c.x * c.z;
                yy += c.y * c.y; yz += c.y * c.z; zz += c.z * c.z;
              });
              const inverse = new THREE.Matrix3().set(xx, xy, xz, xy, yy, yz, xz, yz, zz).invert();
              const correction = error.clone().applyMatrix3(inverse);
              variables.forEach((variable, index) => {
                const delta = THREE.MathUtils.clamp(columns[index].dot(correction) * 0.75, -0.2, 0.2);
                variable.joint.rotation[variable.axis] = THREE.MathUtils.clamp(
                  variable.joint.rotation[variable.axis] + delta, variable.min, variable.max
                );
              });
            }

            scene.updateMatrixWorld(true);
            gripPoint.getWorldPosition(effector);
            const finalError = effector.distanceTo(target);
            if (finalError < best.error) {
              best = {
                error: finalError,
                pose: Object.fromEntries(variables.map((v) => [v.name, v.joint.rotation[v.axis]])),
              };
            }
            if (best.error < 0.08) break;
          }

          return best;
        }

        async function runExtractionSequence() {
          if (!selectedTargetPosition) return;
          const generation = ++sequenceGeneration;
          const current = () => generation === sequenceGeneration;
          setControlsDisabled(true);

          status.textContent = 'RETURNING HOME';
          sequenceState.textContent = '01 / NORMALIZE START POSE';
          await Promise.all([
            tween(joints.base.rotation, { x: 0, y: 0, z: 0 }, 550),
            tween(joints.shoulder.rotation, { x: 0, y: 0, z: 0 }, 550),
            tween(joints.elbow.rotation, { x: 0, y: 0, z: 0 }, 550),
            tween(joints.wrist.rotation, { x: 0, y: 0, z: 0 }, 550),
            tween(gripper.left.rotation, { y: deg(26) }, 550),
            tween(gripper.right.rotation, { y: deg(-26) }, 550),
          ]);
          if (!current()) return;

          const grabTarget = selectedTargetPosition.clone();
          grabTarget.y = 0.42;
          const solution = selectedSolution || solveIK(grabTarget);
          jointVariables().forEach((v) => { v.joint.rotation[v.axis] = 0; });
          scene.updateMatrixWorld(true);

          if (solution.error > 0.22) {
            status.textContent = 'TARGET OUT OF REACH';
            sequenceState.textContent = `IK ERROR ${solution.error.toFixed(2)}m · PLACE TARGET CLOSER`;
            setControlsDisabled(false);
            syncControls();
            return;
          }

          status.textContent = 'IK SOLVED';
          sequenceState.textContent = '02 / BASE ALIGNMENT';
          await tween(joints.base.rotation, { y: solution.pose.base }, 650);
          if (!current()) return;

          status.textContent = 'MOVING TO TARGET';
          sequenceState.textContent = '03 / COORDINATED JOINT MOTION';
          await Promise.all([
            tween(joints.shoulder.rotation, { x: solution.pose.shoulder }, 1200),
            tween(joints.elbow.rotation, { y: solution.pose.elbow }, 1200),
            tween(joints.wrist.rotation, { x: solution.pose.wrist }, 1200),
          ]);
          if (!current()) return;

          scene.updateMatrixWorld(true);
          const reached = new THREE.Vector3();
          gripPoint.getWorldPosition(reached);
          const reachError = reached.distanceTo(grabTarget);
          if (reachError > 0.18) {
            status.textContent = 'APPROACH FAILED';
            sequenceState.textContent = `SAFETY STOP · GRIP ERROR ${reachError.toFixed(2)}m`;
            setControlsDisabled(false);
            syncControls();
            return;
          }

          status.textContent = 'GRIPPING';
          sequenceState.textContent = '04 / TARGET WITHIN GRIP TOLERANCE';
          await Promise.all([
            tween(gripper.left.rotation, { y: deg(2) }, 450),
            tween(gripper.right.rotation, { y: deg(-2) }, 450),
          ]);
          if (!current()) return;
          gripPoint.attach(weedTarget.plant);
          weedTarget.ring.visible = false;

          status.textContent = 'PULLING WEED';
          sequenceState.textContent = '05 / EXTRACTION';
          await tween(joints.shoulder.rotation, { x: solution.pose.shoulder - deg(22) }, 750);
          await wait(350);
          if (!current()) return;

          status.textContent = 'RETURNING HOME';
          sequenceState.textContent = '06 / RETURN HOME';
          weedTarget.plant.visible = false;
          await Promise.all([
            tween(joints.base.rotation, { y: 0 }, 800),
            tween(joints.shoulder.rotation, { x: 0 }, 800),
            tween(joints.elbow.rotation, { y: 0 }, 800),
            tween(joints.wrist.rotation, { x: 0 }, 800),
            tween(gripper.left.rotation, { y: deg(12) }, 800),
            tween(gripper.right.rotation, { y: deg(-12) }, 800),
          ]);
          if (!current()) return;
          weedTarget.target.visible = false;
          selectedTargetPosition = null;
          selectedSolution = null;
          syncControls();
          status.textContent = 'PREPARING NEXT TARGET';
          sequenceState.textContent = 'AUTO IK SEARCH';
          prepareAutoTarget();
        }

        runButton.addEventListener('click', runExtractionSequence);
        document.getElementById('reset-joints').addEventListener('click', () => resetRig(false));

        const loader = new GLTFLoader();
        loader.load(
          'data:model/gltf-binary;base64,__MODEL_DATA__',
          (gltf) => {
            const robot = gltf.scene;
            buildJointRig(robot);
            let box = new THREE.Box3().setFromObject(robot);
            const size = box.getSize(new THREE.Vector3());
            const scale = 4.4 / Math.max(size.x, size.y, size.z);
            robot.scale.setScalar(scale);
            box = new THREE.Box3().setFromObject(robot);
            const center = box.getCenter(new THREE.Vector3());
            robot.position.set(-center.x, -box.min.y, -center.z);
            scene.add(robot);
            robot.updateMatrixWorld(true);
            joints.base.getWorldPosition(workspaceCenter);
            workspaceCenter.y = 0;

            weedTarget = buildWeedTarget();

            const fittedBox = new THREE.Box3().setFromObject(robot);
            const fittedSize = fittedBox.getSize(new THREE.Vector3());
            const fittedCenter = fittedBox.getCenter(new THREE.Vector3());
            const maxDim = Math.max(fittedSize.x, fittedSize.y, fittedSize.z);
            const distance = ((maxDim / 2) / Math.tan(deg(camera.fov / 2))) * 1.65;
            controls.target.copy(fittedCenter);
            camera.position.set(
              fittedCenter.x + distance * 0.85,
              fittedCenter.y + distance * 0.35,
              fittedCenter.z + distance * 0.85
            );
            camera.near = Math.max(0.01, distance / 100);
            camera.far = distance * 20;
            camera.updateProjectionMatrix();
            controls.minDistance = distance * 0.45;
            controls.maxDistance = distance * 3;
            controls.update();
            resetRig();
            status.textContent = 'SEARCHING TARGET';
            sequenceState.textContent = 'AUTO IK SEARCH';
            setTimeout(prepareAutoTarget, 50);
            loading.classList.add('done');
            setTimeout(() => loading.remove(), 400);
          },
          (event) => {
            if (event.total) loading.textContent = `LOADING MODEL · ${Math.round(event.loaded / event.total * 100)}%`;
          },
          (error) => {
            console.error(error);
            status.textContent = 'LOAD ERROR';
            loading.textContent = 'MODEL COULD NOT BE LOADED';
          }
        );

        function resize() {
          const width = container.clientWidth;
          const height = container.clientHeight;
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
        }
        window.addEventListener('resize', resize);

        const clock = new THREE.Clock();
        function animate() {
          requestAnimationFrame(animate);
          const dt = Math.min(50, clock.getDelta() * 1000);
          updateTweens(dt);
          if (weedTarget?.target.visible && weedTarget.ring.visible) {
            weedTarget.pulse += dt / 1000;
            const pulse = 1 + Math.sin(weedTarget.pulse * 4) * 0.08;
            weedTarget.ring.scale.setScalar(pulse);
          }
          controls.update();
          renderer.render(scene, camera);
        }
        animate();
      </script>
    </body>
    </html>
    """.replace("__MODEL_DATA__", model_data)

    components.html(viewer_html, height=640, scrolling=False)


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
        current_detection = st.session_state.detection
        current_targets = current_detection["targets"] if current_detection else None
        code_slot.html(
            code_panel(
                uploaded_file.name,
                current_targets,
                raw_count=current_detection.get("raw_count") if current_detection else None,
                outside_roi_count=current_detection.get("outside_roi_count") if current_detection else None,
                error=st.session_state.detection_error,
            )
        )

    if detect_clicked:
        code_slot.html(code_panel(uploaded_file.name, status="running"))
        st.session_state.detection_error = None
        try:
            with st.spinner("Running inference…"):
                result = run_inference(image, get_api_key())
                raw_predictions = result.get("predictions", [])
                raw_count = len(raw_predictions)
                img_width, img_height = image.size

                # The camera is fixed above the pot. Discard detections whose
                # centre lies in the upper background, outside the pot workspace.
                predictions = [
                    pred
                    for pred in raw_predictions
                    if pred["y"] >= img_height * ROI_TOP_RATIO
                ]
                outside_roi_count = raw_count - len(predictions)

                output, targets = annotate_image(image, predictions)
                st.session_state.detection = {
                    "output": output,
                    "targets": targets,
                    "predictions": predictions,
                    "raw_count": raw_count,
                    "outside_roi_count": outside_roi_count,
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

st.html('<div class="section-label">04 / <b>Robotic arm digital twin</b></div>')
render_robot_viewer(os.path.join(os.path.dirname(__file__), "assets", "robot_arm.glb"))

"""
AI Blind Assist System
======================
Real-time hazard and obstacle detection for visually impaired users.
Pipeline (as described in the article):
  * YOLOv8   - object detection
  * MiDaS    - monocular *relative* depth estimation
  * distance = 1000 / depth  (empirical, NOT metric - see article, Limitations)
  * three-tier risk classification:
        high   < 2 m   -> red box,    voice warning (pyttsx3)
        medium 2-5 m   -> yellow box, visual warning
        safe   > 5 m   -> green box,  monitoring only
  * cooldown-controlled voice alerts (no repeated warnings for the same object)
  * task-level parallelism: YOLOv8 and MiDaS run as two concurrent processes
    (multiprocessing); voice output runs in a separate thread (threading).
"""
import argparse
import multiprocessing as mp
import queue
import threading
import time
import cv2
import numpy as np
# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HIGH_RISK_M = 2.0          # < 2 m  -> high risk
MEDIUM_RISK_M = 5.0        # 2-5 m  -> medium risk, > 5 m -> safe
DISTANCE_K = 1000.0        # distance = DISTANCE_K / depth (empirical formula)
COOLDOWN_S = 5.0           # min. seconds between voice alerts for same object
COLORS = {                 # BGR
    "high": (0, 0, 255),
    "medium": (0, 255, 255),
    "safe": (0, 255, 0),
}
# --------------------------------------------------------------------------- #
# Risk logic
# --------------------------------------------------------------------------- #
def classify_risk(distance_m: float) -> str:
    """Three-tier, distance-based risk classification."""
    if distance_m < HIGH_RISK_M:
        return "high"
    if distance_m <= MEDIUM_RISK_M:
        return "medium"
    return "safe"
def estimate_distance(depth: np.ndarray, box) -> float:
    """Approximate distance from the MiDaS value at the bounding-box centre.
    MiDaS returns inverse relative depth (larger value = closer), so the
    empirical formula distance = 1000 / depth is used. A small patch around
    the centre is median-filtered to reduce noise.
    """
    x1, y1, x2, y2 = box
    h, w = depth.shape[:2]
    cx = int(np.clip((x1 + x2) // 2, 0, w - 1))
    cy = int(np.clip((y1 + y2) // 2, 0, h - 1))
    r = 2
    patch = depth[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
    value = float(np.median(patch))
    if value <= 1e-6:
        return float("inf")
    return DISTANCE_K / value
# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class YoloDetector:
    """YOLOv8 object detector. Returns plain tuples (picklable)."""
    def __init__(self, weights: str = "yolov8n.pt"):
        import torch
        from ultralytics import YOLO
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weights)
        self(np.zeros((480, 640, 3), np.uint8))          # warm-up
    def __call__(self, frame):
        result = self.model(frame, verbose=False, device=self.device)[0]
        detections = []
        for b in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            detections.append((x1, y1, x2, y2, float(b.conf[0]),
                               result.names[int(b.cls[0])]))
        return detections
class MidasEstimator:
    """MiDaS (small) monocular relative depth estimator."""
    def __init__(self):
        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        self(np.zeros((480, 640, 3), np.uint8))          # warm-up
    def __call__(self, frame) -> np.ndarray:
        torch = self.torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch = self.transform(rgb).to(self.device)
        with torch.no_grad():
            pred = self.model(batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=rgb.shape[:2],
                mode="bicubic", align_corners=False).squeeze()
        return pred.cpu().numpy().astype(np.float32)
# --------------------------------------------------------------------------- #
# Worker processes (one execution unit per model)
# --------------------------------------------------------------------------- #
def yolo_worker(frame_q, result_q, ready, weights):
    detector = YoloDetector(weights)
    ready.set()
    while True:
        item = frame_q.get()
        if item is None:
            break
        frame_id, frame = item
        t0 = time.perf_counter()
        detections = detector(frame)
        result_q.put((frame_id, detections, time.perf_counter() - t0))
def midas_worker(frame_q, result_q, ready):
    estimator = MidasEstimator()
    ready.set()
    while True:
        item = frame_q.get()
        if item is None:
            break
        frame_id, frame = item
        t0 = time.perf_counter()
        depth = estimator(frame)
        result_q.put((frame_id, depth, time.perf_counter() - t0))
# --------------------------------------------------------------------------- #
# Voice alerts (separate thread, cooldown-controlled)
# --------------------------------------------------------------------------- #
class VoiceAlert:
    def __init__(self, cooldown: float = COOLDOWN_S):
        self.cooldown = cooldown
        self.last_alert = {}
        self.messages = queue.Queue(maxsize=5)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    def _run(self):
        import pyttsx3
        engine = pyttsx3.init()          # created inside the thread that uses it
        while True:
            message = self.messages.get()
            if message is None:
                break
            engine.say(message)
            engine.runAndWait()
    def alert(self, label: str, distance_m: float):
        now = time.time()
        if now - self.last_alert.get(label, 0.0) < self.cooldown:
            return                       # cooldown: suppress repeated alert
        try:
            self.messages.put_nowait(
                f"Warning! {label} ahead, about {distance_m:.1f} meters")
            self.last_alert[label] = now
        except queue.Full:
            pass
    def close(self):
        self.messages.put(None)
# --------------------------------------------------------------------------- #
# Frame processing / drawing
# --------------------------------------------------------------------------- #
def process_frame(frame, detections, depth, voice=None):
    """Combine detections + depth, classify risk, draw boxes, trigger alerts."""
    for x1, y1, x2, y2, _conf, label in detections:
        distance = estimate_distance(depth, (x1, y1, x2, y2))
        risk = classify_risk(distance)
        color = COLORS[risk]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {distance:.1f}m [{risk}]"
        cv2.putText(frame, text, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if risk == "high" and voice is not None:
            voice.alert(label, distance)
    return frame
class Stats:
    def __init__(self):
        self.t_yolo, self.t_midas, self.latency = [], [], []
        self.t_start = self.t_end = None
    def add(self, t_yolo, t_midas, latency):
        self.t_yolo.append(t_yolo)
        self.t_midas.append(t_midas)
        self.latency.append(latency)
    @property
    def n(self):
        return len(self.latency)
    def avg_yolo(self):
        return float(np.mean(self.t_yolo)) if self.t_yolo else 0.0
    def avg_midas(self):
        return float(np.mean(self.t_midas)) if self.t_midas else 0.0
    def summary(self):
        elapsed = self.t_end - self.t_start
        return {
            "frames": self.n,
            "yolo_s": self.avg_yolo(),
            "midas_s": self.avg_midas(),
            "total_s": float(np.mean(self.latency)) if self.latency else 0.0,
            "fps": self.n / elapsed if elapsed > 0 else 0.0,
        }
def open_source(source: str):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap
def show(frame, fps_text, window="AI Blind Assist System"):
    """Draw multi-line FPS info (Parallel + Sequential) on the frame."""
    y0 = 24
    for i, line in enumerate(fps_text.split("\n")):
        cv2.putText(frame, line, (10, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(window, frame)
    return cv2.waitKey(1) & 0xFF != ord("q")
def make_fps_text(mode: str, stats: Stats, elapsed: float) -> str:
    """
    Build a multi-line string that shows BOTH sequential and parallel FPS
    (measured + estimated) so the user can instantly see the difference.
    """
    n = stats.n
    if n == 0 or elapsed <= 0:
        return f"{mode}\nFPS: --"
    current_fps = n / elapsed
    avg_y = stats.avg_yolo()
    avg_m = stats.avg_midas()
    # Ideal sequential latency ≈ YOLO + MiDaS
    seq_est = 1.0 / (avg_y + avg_m) if (avg_y + avg_m) > 1e-6 else 0.0
    # Ideal parallel latency ≈ max(YOLO, MiDaS)  (perfect overlap)
    par_est = 1.0 / max(avg_y, avg_m) if max(avg_y, avg_m) > 1e-6 else 0.0
    if mode.lower().startswith("parallel"):
        # Measured = parallel throughput; estimated sequential for comparison
        speedup = current_fps / seq_est if seq_est > 1e-6 else 0.0
        return (f"Parallel FPS: {current_fps:.1f}\n"
                f"Seq. est. FPS: {seq_est:.1f}\n"
                f"Speedup: {speedup:.2f}x")
    else:
        # Measured = sequential; estimated parallel for comparison
        speedup = par_est / current_fps if current_fps > 1e-6 else 0.0
        return (f"Sequential FPS: {current_fps:.1f}\n"
                f"Par. est. FPS: {par_est:.1f}\n"
                f"Speedup: {speedup:.2f}x")
# --------------------------------------------------------------------------- #
# Sequential pipeline (baseline)
# --------------------------------------------------------------------------- #
def run_sequential(args, voice=None, display=True):
    detector = YoloDetector(args.weights)
    estimator = MidasEstimator()
    cap = open_source(args.source)
    stats = Stats()
    stats.t_start = time.perf_counter()
    try:
        while args.frames == 0 or stats.n < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            detections = detector(frame)
            t1 = time.perf_counter()
            depth = estimator(frame)
            t2 = time.perf_counter()
            stats.add(t1 - t0, t2 - t1, t2 - t0)
            frame = process_frame(frame, detections, depth, voice)
            if display:
                elapsed = time.perf_counter() - stats.t_start
                fps_text = make_fps_text("Sequential", stats, elapsed)
                if not show(frame, fps_text):
                    break
    finally:
        stats.t_end = time.perf_counter()
        cap.release()
        if display:
            cv2.destroyAllWindows()
    return stats
# --------------------------------------------------------------------------- #
# Parallel pipeline: YOLOv8 || MiDaS (two processes), main process = capture,
# synchronisation, risk classification, drawing.
# --------------------------------------------------------------------------- #
def run_parallel(args, voice=None, display=True):
    ctx = mp.get_context("spawn")
    yolo_in, midas_in = ctx.Queue(maxsize=args.inflight), ctx.Queue(maxsize=args.inflight)
    yolo_out, midas_out = ctx.Queue(), ctx.Queue()
    ready = [ctx.Event(), ctx.Event()]
    procs = [
        ctx.Process(target=yolo_worker, args=(yolo_in, yolo_out, ready[0], args.weights), daemon=True),
        ctx.Process(target=midas_worker, args=(midas_in, midas_out, ready[1]), daemon=True),
    ]
    for p in procs:
        p.start()
    for ev in ready:                       # wait for model load + warm-up
        ev.wait()
    cap = open_source(args.source)
    stats = Stats()
    pending = {}                           # frame_id -> state
    next_submit = next_done = 0
    eof = False
    stats.t_start = time.perf_counter()
    try:
        while True:
            # 1) submit new frames (up to `inflight` frames in the pipeline)
            while (not eof and len(pending) < args.inflight
                   and (args.frames == 0 or next_submit < args.frames)):
                ok, frame = cap.read()
                if not ok:
                    eof = True
                    break
                pending[next_submit] = {"frame": frame, "t0": time.perf_counter()}
                yolo_in.put((next_submit, frame))
                midas_in.put((next_submit, frame))
                next_submit += 1
            if args.frames and next_submit >= args.frames:
                eof = True
            if not pending and eof:
                break
            # 2) collect results from both workers
            got = False
            for q, key, tkey in ((yolo_out, "det", "t_yolo"), (midas_out, "depth", "t_midas")):
                try:
                    while True:
                        frame_id, result, dt = q.get_nowait()
                        pending[frame_id][key] = result
                        pending[frame_id][tkey] = dt
                        got = True
                except queue.Empty:
                    pass
            if not got:
                time.sleep(0.001)
            # 3) finish frames in order once both results are available
            while next_done in pending and "det" in pending[next_done] and "depth" in pending[next_done]:
                s = pending.pop(next_done)
                stats.add(s["t_yolo"], s["t_midas"], time.perf_counter() - s["t0"])
                frame = process_frame(s["frame"], s["det"], s["depth"], voice)
                next_done += 1
                if display:
                    elapsed = time.perf_counter() - stats.t_start
                    fps_text = make_fps_text("Parallel", stats, elapsed)
                    if not show(frame, fps_text):
                        eof, pending = True, {}
                        break
    finally:
        stats.t_end = time.perf_counter()
        for q in (yolo_in, midas_in):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for p in procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        cap.release()
        if display:
            cv2.destroyAllWindows()
    return stats
# --------------------------------------------------------------------------- #
# Benchmark: reproduces the quantities reported in Table 1 of the article
# --------------------------------------------------------------------------- #
def run_benchmark(args):
    print("Running sequential baseline ...")
    seq = run_sequential(args, voice=None, display=False).summary()
    print("Running parallel pipeline ...")
    par = run_parallel(args, voice=None, display=False).summary()
    print("\nProcessing   YOLO (s)  MiDaS (s)  Total (s)   FPS")
    for name, s in (("Sequential", seq), ("Parallel", par)):
        print(f"{name:<12} {s['yolo_s']:.3f}     {s['midas_s']:.3f}      "
              f"{s['total_s']:.3f}      {s['fps']:.1f}")
    print(f"\nThroughput speedup  S_FPS  = FPS_par / FPS_seq = {par['fps'] / seq['fps']:.2f}x")
    print(f"Latency speedup     S_time = T_seq / T_par     = {seq['total_s'] / par['total_s']:.2f}x")
    print("(Use --inflight 1 to measure pure per-frame latency without pipeline overlap.)")
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="AI Blind Assist System")
    ap.add_argument("--mode", choices=["parallel", "sequential", "benchmark"], default="parallel")
    ap.add_argument("--source", default="0", help="webcam index or video file path")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--frames", type=int, default=0, help="max frames (0 = until stream ends / 'q')")
    ap.add_argument("--inflight", type=int, default=2, help="frames simultaneously in the parallel pipeline")
    ap.add_argument("--cooldown", type=float, default=COOLDOWN_S, help="voice-alert cooldown, seconds")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if args.mode == "benchmark":
        if args.frames == 0:
            args.frames = 200
        run_benchmark(args)
        return
    voice = None if args.no_voice else VoiceAlert(args.cooldown)
    runner = run_parallel if args.mode == "parallel" else run_sequential
    try:
        stats = runner(args, voice=voice, display=not args.no_show)
    finally:
        if voice:
            voice.close()
    s = stats.summary()
    print(f"{args.mode}: {s['frames']} frames, {s['fps']:.2f} FPS, "
          f"avg per-frame time {s['total_s']:.3f} s")
    # Qo'shimcha taqqoslash (farqni aniq ko'rish uchun)
    avg_y = s["yolo_s"]
    avg_m = s["midas_s"]
    seq_est = 1.0 / (avg_y + avg_m) if (avg_y + avg_m) > 1e-6 else 0.0
    par_est = 1.0 / max(avg_y, avg_m) if max(avg_y, avg_m) > 1e-6 else 0.0
    if args.mode == "parallel":
        print(f"  → Sequential estimate: {seq_est:.2f} FPS")
        print(f"  → Speedup vs sequential: {s['fps'] / seq_est:.2f}x" if seq_est > 0 else "")
    else:
        print(f"  → Parallel estimate:   {par_est:.2f} FPS")
        print(f"  → Potential speedup:   {par_est / s['fps']:.2f}x" if s['fps'] > 0 else "")
if __name__ == "__main__":
    mp.freeze_support()          # required for Windows executables
    main()
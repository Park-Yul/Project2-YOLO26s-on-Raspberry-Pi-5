# door_masker.py  (마스커 Pi 10.10.17.105: 판정 + 마스크 그리기 + X11 화면 출력)
# ─────────────────────────────────────────────────────────────────────────────
# 4대 구성
#   마스터 : 카메라 -> 작업자 2대에게 프레임 분배                         (door_master.py)
#   작업자 : 추론(yolo26s-seg) + 사람 윤곽 좌표 추출 -> 마스커로 전송      (door_worker.py)
#   마스커 : 순서 정렬 + 추적 + IN/OUT 판정 + 마스크 그리기 + X11 출력     (이 파일)
#
# 하는 일
#   1) 작업자 2대가 보낸 [JPEG + 박스 + 윤곽 좌표] 를 프레임 번호 순서대로 정렬
#   2) ByteTrack 추적 + A/B 구역 IN/OUT 판정 (마스크 무게중심 / 마스크 가림 기준)
#   3) 반투명 마스크·박스·라벨·A 구역·학생 수 그리기
#   4) X11 창으로 출력 (cv2.imshow). 화면 출력은 판정과 다른 스레드라 화면이 느려도 카운트는 안 밀림
#   5) (나중에 붙일 웹 앱용) 영상/숫자 파일 저장 + 제어 포트 6200  -> APP_LINK 로 켜고 끔
#
# 실행 (작업자들이 마스커로 접속하므로 마스커를 먼저 켜는 것을 권장)
#   source ~/work/yolo-env/bin/activate && python door_masker.py
#   X11 로 보려면: MobaXterm 등 X11 포워딩으로 SSH 접속하거나, 마스커 Pi 모니터에서 실행
# 키 (X11 창에서): q 종료, r 학생 수를 TOTAL_START 로 리셋 (IN/OUT 0)
# 필요한 패키지: opencv, numpy, ultralytics(ByteTrack 용. 모델 파일은 필요 없음)
#
# 통신 형식 (TCP, 작업자 -> 마스커)
#   [프레임번호 4B][실행ID 4B][촬영시각 8B double][추론시간ms 4B float][사람 수 N 4B][JPEG길이 4B]
#   [JPEG][N x 6 float32 (x1,y1,x2,y2,conf,cls)][N x 점개수 uint32][모든 윤곽 점 (x,y) float32]
# 제어 포트 (APP_LINK=True 일 때, JSON 한 줄)
#   {"cmd":"get"} / {"cmd":"set_total","value":20} / {"cmd":"reset"}  ->  {"ok":true,"total":..,"in":..,"out":..,"fps":..}
# ─────────────────────────────────────────────────────────────────────────────

import os

# ---------------- CPU 비교 실험 스위치 ----------------
# BUSY_WAIT = False : 수학 라이브러리 스레드 1개 + 기다릴 때 잠듦   (기본. CPU 절약)
# BUSY_WAIT = True  : 스레드 수 기본값(코어 수) + 기다릴 때 계속 돎 (예전 마스터와 같은 상태)
# 판정 결과는 둘 다 같음. 로그의 'CPU xx%' 와 '판정+그리기 xxms' 를 비교하면 됨.
# ※ 프로그램 시작할 때 한 번만 적용됨 -> 바꾸면 다시 실행
BUSY_WAIT = False

_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME")
_outside = {k: os.environ[k] for k in _THREAD_ENV if k in os.environ}   # 터미널에서 export 된 값 (실험을 망칠 수 있음)
if BUSY_WAIT:
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.pop(_k, None)                         # 스레드 수 제한 없음 (코어 수만큼)
    os.environ["OMP_WAIT_POLICY"] = "ACTIVE"             # 기다릴 때 잠들지 않고 돎
    os.environ["GOMP_SPINCOUNT"] = "INFINITE"            # GNU OpenMP
    os.environ["KMP_BLOCKTIME"] = "infinite"             # LLVM OpenMP
else:
    for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0", "KMP_BLOCKTIME": "0"}.items():
        os.environ[_k] = _v

import inspect
import json
import socket
import struct
import sys
import threading
import time
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker

if not BUSY_WAIT:
    torch.set_num_threads(1)
    cv2.setNumThreads(2)

# ---------------- 통신 ----------------
PORT = 6100                       # 작업자들이 접속하는 포트
REORDER_TIMEOUT = 0.6             # 다음 번호 프레임이 이 시간(초) 안에 안 오면 건너뜀 (작업자 끊김 등)
TOTAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "door_total.json")  # 현재 인원 저장 (재시작해도 유지)

# ---------------- X11 화면 출력 ----------------
SHOW = True                       # X11 창 출력. DISPLAY 가 없으면(일반 SSH) 자동으로 끔
SHOW_MAX_FPS = 15                 # 화면 출력 최대 속도 (X11 전송량 조절. 판정 속도와는 무관)
SHOW_SCALE = 1.0                  # 화면 크기 배율 (X11 이 버벅이면 0.75 / 0.5 로 줄이기)
WINDOW_NAME = "door_masker"

# ---------------- 웹 앱 연동 (app.py 는 나중에 수정 예정) ----------------
APP_LINK = True                   # True: 아래 파일 저장 + 제어 포트 열기 / False: 끄기 (CPU 절약)
CONTROL_PORT = 6200               # 웹 앱이 인원 설정/조회하는 포트
FRAME_PATH = "/tmp/door_frame.jpg"     # 웹 앱이 읽는 영상
STATS_PATH = "/tmp/door_stats.json"    # 웹 앱이 읽는 인원/IN/OUT/FPS
OUT_JPEG_QUALITY = 75

# ---------------- 그리기 ----------------
MASK_ALPHA = 0.45
DRAW_LABEL = True                 # 사람 라벨 "person: 0.91 id7 B>B"
DRAW_HUD = True                   # 영상 왼쪽 위 Total / IN / OUT / FPS 글자
PALETTE = [
    (255, 178, 29), (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255), (52, 69, 147),
    (100, 115, 255), (0, 24, 236), (132, 56, 255), (82, 0, 133), (203, 56, 255),
    (255, 149, 200), (255, 55, 199),
]

# ---------------- 추적 (ByteTrack) ----------------
TRACK_ARGS = SimpleNamespace(
    tracker_type="bytetrack",
    track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25,
    track_buffer=30, match_thresh=0.8, fuse_score=True,
)
TRACK_FPS = 30

# ---------------- A(문 쪽) 구역 ----------------
A_LEFT_RATIO = 0.40
A_RIGHT_RATIO = 0.91
A_BOTTOM_RATIO = 0.60
A_TOP_RATIO = None
R_BL_RATIO = 0.17
R_BR_RATIO = 0.06
USE_FOOT_Y = False

# ---------------- 판정 (시간은 초 단위, 마스터 카메라 촬영시각 기준) ----------------
BAND_RATIO = 0.025
MERGE_RATIO = 0.19
LOST_SEC = 2.5                    # 추적기(track_buffer 30프레임 = 초당 16장에서 약 1.8초)보다 길게. 짧으면 되살아난 ID 의 출신이 지워짐
REAPPEAR_SEC = 2.5                # 위와 같은 이유 (가려진 ID 에 적용되는 값)
OCCLUDE_RATIO = 0.3
OCCLUDER_NEAR = 1.0
JUDGE_BY_MASK = True
MASK_MIN_FILL = 0.25
MASK_OCCLUDE_RATIO = 0.2
MASK_SCALE = 0.5

# ---------------- 출신 판정 보완 (가상 상황 실험으로 고른 값. door_sim.py 로 다시 비교 가능) ----------------
# LINK_GUARD: B 에서 새로 생긴 ID 가, 안 보이는 다른 ID 의 출신을 물려받는 것을 제한
#   새로 온 사람이 가려져 있는 사람의 출신(B)을 가져가 버리면, 그 사람은 A 에서 다시 잡힐 때 출신 A 가 되어 나가도 안 세짐
#   0: 제한 없음 (예전 방식)
#   1: '출신 B 이고 B 에서 사라진 ID' 의 것은 물려받지 않음
#   2: '출신 B 인 ID' 의 것은 어디서 사라졌든 물려받지 않음   <- Total 이 실제보다 적게 나오기 시작하면 1 로 낮추기
LINK_GUARD = 2
# BIRTH_BAND_RATIO: A 안쪽 띠의 폭 (화면 너비 비율. 0 이면 끔). 왼쪽 선·아래쪽 선에서 이 거리 안에서
#   '연결 없이 새로 생긴' ID 는 출신을 B 로 봄 (경계를 넘는 장면이 가려져 A 에서 처음 잡힌 교실 사람)
#   문으로 들어온 사람은 문 위치에서 처음 잡히므로, 띠가 문까지 닿지 않게 할 것
BIRTH_BAND_RATIO = 0.05
DRAW_BAND = True                  # 띠의 안쪽 경계(보조선)를 화면에 그림

TOTAL_START = 3                  # door_total.json 이 없을 때의 시작 인원
DEBUG = True

HDR = struct.Struct("!IIdfII")    # seq, run_id, t_cap, infer_ms, n, jpeg_len


def log(*a):
    print(*a, flush=True)


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("연결 끊김")
        buf.extend(chunk)
    return bytes(buf)


def write_atomic(path, data):
    try:
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
    except OSError:
        pass


# ================= 인원 상태 (저장/불러오기) =================
total, in_cnt, out_cnt = TOTAL_START, 0, 0
try:
    with open(TOTAL_FILE) as _f:
        total = int(json.load(_f)["total"])
    log(f"[마스커] 저장된 현재 인원 {total}명 불러옴 ({os.path.basename(TOTAL_FILE)})")
except (OSError, ValueError, KeyError):
    log(f"[마스커] 저장된 인원 없음 -> 시작 인원 {total}명")


def save_total():
    write_atomic(TOTAL_FILE, json.dumps({"total": int(total), "ts": time.time()}).encode("utf-8"))


# ================= 1. 작업자 수신 (작업자마다 스레드 1개) =================
buf_cond = threading.Condition()
buffer = {}                       # seq -> 패킷
cur_run = None                    # 현재 마스터 실행ID
old_runs = set()                  # 지나간 실행ID (늦게 온 옛 프레임 무시용)
need_reset = False                # 마스터 재시작 -> 추적 상태 초기화
expect = None                     # 다음에 처리할 프레임 번호
worker_ms = {}                    # 작업자 "IP:포트" -> 최근 추론 ms
workers_up = set()


def worker_conn(conn, addr):
    global cur_run, need_reset, expect
    ip = f"{addr[0]}:{addr[1]}"          # 같은 IP 에 작업자가 둘이어도 구분되게 포트까지
    workers_up.add(ip)
    log(f"[마스커] 작업자 연결됨: {addr[0]}")
    try:
        while True:
            seq, run_id, t_cap, ms, n, jlen = HDR.unpack(recv_exact(conn, HDR.size))
            jpg = recv_exact(conn, jlen)
            if n:
                arr = np.frombuffer(recv_exact(conn, n * 24), np.float32).reshape(n, 6)
                counts = np.frombuffer(recv_exact(conn, n * 4), ">u4").astype(np.int64)
                tp = int(counts.sum())
                pts = (np.frombuffer(recv_exact(conn, tp * 8), np.float32).reshape(-1, 2)
                       if tp else np.zeros((0, 2), np.float32))
                polys = np.split(pts, np.cumsum(counts)[:-1])
            else:
                arr, polys = np.zeros((0, 6), np.float32), []
            worker_ms[ip] = ms
            with buf_cond:
                if run_id in old_runs:
                    continue
                if run_id != cur_run:                 # 마스터가 새로 켜짐
                    if cur_run is not None:
                        old_runs.add(cur_run)
                        log(f"[마스커] 마스터 재시작 감지 -> 프레임 번호/추적 초기화 (인원 수는 유지)")
                        need_reset = True
                    cur_run, expect = run_id, None
                    buffer.clear()
                if expect is None or seq >= expect:
                    buffer[seq] = (t_cap, jpg, arr, polys)
                    buf_cond.notify_all()
    except (ConnectionError, OSError, ValueError) as e:
        log(f"[마스커] 작업자 연결 끊김: {addr[0]} ({e})")
    finally:
        workers_up.discard(ip)
        worker_ms.pop(ip, None)
        conn.close()


def accept_loop():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(4)
    log(f"[마스커] {PORT} 포트에서 작업자 접속 대기 중...")
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=worker_conn, args=(conn, addr), daemon=True).start()


# ================= 2. 인원 설정 (키 r / 웹 앱 제어 포트) =================
ctrl_lock = threading.Lock()
ctrl_cmd = None                   # 처리 루프가 적용할 명령 ("set", 값)
ctrl_done = threading.Event()
fps_now = 0.0


def control_loop():
    global ctrl_cmd
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", CONTROL_PORT))
    srv.listen(4)
    log(f"[마스커] 제어 포트 {CONTROL_PORT} 대기 (웹에서 인원 설정)")
    while True:
        conn, addr = srv.accept()
        with conn:
            try:
                conn.settimeout(3)
                req = json.loads(conn.makefile("r", encoding="utf-8").readline() or "{}")
                cmd = req.get("cmd")
                if cmd in ("set_total", "reset"):
                    v = TOTAL_START if cmd == "reset" else int(req["value"])
                    if v < 0:
                        raise ValueError("0 이상이어야 함")
                    ctrl_done.clear()
                    with ctrl_lock:
                        ctrl_cmd = v
                    if not ctrl_done.wait(2):
                        raise TimeoutError("처리 루프 응답 없음")
                elif cmd != "get":
                    raise ValueError(f"알 수 없는 명령: {cmd}")
                resp = {"ok": True, "total": total, "in": in_cnt, "out": out_cnt, "fps": round(fps_now, 1)}
            except Exception as e:
                resp = {"ok": False, "error": str(e)}
            try:
                conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
            except OSError:
                pass


# ================= 3. 구역 / 판정 =================
FRAME_W = FRAME_H = None
AL = AR = AB = AT = R_BL = R_BR = BAND = None
MERGE_DIST = S = LW = BBAND = None


def set_geometry(w, h):
    global FRAME_W, FRAME_H, AL, AR, AB, AT, R_BL, R_BR, BAND, MERGE_DIST, S, LW, BBAND
    FRAME_W, FRAME_H = w, h
    AL, AR = int(w * A_LEFT_RATIO), int(w * A_RIGHT_RATIO)
    AB = int(h * A_BOTTOM_RATIO)
    AT = int(h * A_TOP_RATIO) if A_TOP_RATIO is not None else None
    R_BL, R_BR = int(h * R_BL_RATIO), int(h * R_BR_RATIO)
    BAND = max(1, int(w * BAND_RATIO))
    MERGE_DIST = w * MERGE_RATIO
    BBAND = int(w * BIRTH_BAND_RATIO)
    S = w / 640
    LW = max(1, round(2 * S))
    log(f"[마스커] 영상 크기 {w}x{h}")


def a_dist(x, y):
    dl, dr, db = x - AL, AR - x, AB - y
    dt = (y - AT) if AT is not None else float("inf")
    if R_BL > 0 and dl < R_BL and db < R_BL:
        return R_BL - ((R_BL - dl) ** 2 + (R_BL - db) ** 2) ** 0.5
    if R_BR > 0 and dr < R_BR and db < R_BR:
        return R_BR - ((R_BR - dr) ** 2 + (R_BR - db) ** 2) ** 0.5
    return min(dl, dr, db, dt)


def edge_dist(x, y):
    """A 의 왼쪽 선·아래쪽 선까지의 거리 (A 안쪽이 +). 오른쪽 선과 위쪽은 보지 않음."""
    dl, db = x - AL, AB - y
    if R_BL > 0 and dl < R_BL and db < R_BL:
        return R_BL - ((R_BL - dl) ** 2 + (R_BL - db) ** 2) ** 0.5
    return min(dl, db)


def get_zone(x, y, prev):
    d = a_dist(x, y)
    if d > BAND:
        return "A"
    if d < -BAND:
        return "B"
    if prev is not None:
        return prev
    return "A" if d > 0 else "B"


def new_tracker():
    if "frame_rate" in inspect.signature(BYTETracker.__init__).parameters:
        return BYTETracker(TRACK_ARGS, frame_rate=TRACK_FPS)
    TRACK_ARGS.frame_rate = TRACK_FPS
    return BYTETracker(TRACK_ARGS)


tracker = new_tracker()
origin_of, zone_of, last_seen, last_pos, occluder_of, cur_box = {}, {}, {}, {}, {}, {}


def reset_tracking():
    global tracker
    tracker = new_tracker()
    for d in (origin_of, zone_of, last_seen, last_pos, occluder_of, cur_box):
        d.clear()


def visible_anchor(occ):
    seen = set()
    while occ is not None and occ not in seen:
        if occ in cur_box:
            return occ
        seen.add(occ)
        occ = occluder_of.get(occ)
    return None


def finalize(tid, reason):
    global total, out_cnt
    o, z = origin_of.pop(tid, None), zone_of.pop(tid, None)
    for d in (last_seen, last_pos, occluder_of):
        d.pop(tid, None)
    result = None
    if o == "B" and z == "A":
        out_cnt += 1
        total = max(0, total - 1)
        save_total()
        result = "OUT"
    if DEBUG:
        tag = f"[확정 {result}]" if result else "[무효]  "
        log(f"{tag} id{tid}  {o}->{z}  ({reason})  학생 수={total}")


def judge_point(box, poly):
    """판정 점 (cx, cy, py, 마스크사용여부). 마스크가 너무 작으면 박스 중심."""
    x1, y1, x2, y2 = box
    bx, by = (x1 + x2) / 2, (y1 + y2) / 2
    if JUDGE_BY_MASK and len(poly) >= 3:
        m = cv2.moments(poly.reshape(-1, 1, 2).astype(np.float32))
        if m["m00"] >= MASK_MIN_FILL * max(1.0, (x2 - x1) * (y2 - y1)):
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            return cx, cy, (float(poly[:, 1].max()) if USE_FOOT_Y else cy), True
    return bx, by, (y2 if USE_FOOT_Y else by), False


def occlusion_map(ids, boxes, tpolys):
    """{ti: 가장 많이 가린 사람}. 마스크 있으면 '내 박스 중 상대 마스크 비율', 없으면 박스 겹침."""
    masks = {}
    if JUDGE_BY_MASK:
        h, w = int(FRAME_H * MASK_SCALE), int(FRAME_W * MASK_SCALE)
        for tj, poly in zip(ids, tpolys):
            if len(poly) >= 3:
                mk = np.zeros((h, w), np.uint8)
                cv2.fillPoly(mk, [(poly * MASK_SCALE).astype(np.int32)], 1)
                masks[tj] = mk
    occ = {}
    for i, (ti, (ax1, ay1, ax2, ay2)) in enumerate(zip(ids, boxes)):
        area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        sx1, sy1 = max(0, int(ax1 * MASK_SCALE)), max(0, int(ay1 * MASK_SCALE))
        sx2, sy2 = int(ax2 * MASK_SCALE), int(ay2 * MASK_SCALE)
        s_area = max(1, (sx2 - sx1) * (sy2 - sy1))
        best_s, best_t = 0.0, None
        for j, (tj, (bx1, by1, bx2, by2)) in enumerate(zip(ids, boxes)):
            if i == j:
                continue
            if tj in masks:
                s = (np.count_nonzero(masks[tj][sy1:sy2, sx1:sx2]) / s_area) / MASK_OCCLUDE_RATIO
            else:
                iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                ih = max(0.0, min(ay2, by2) - max(ay1, by1))
                s = (iw * ih / area) / OCCLUDE_RATIO
            if s > best_s:
                best_s, best_t = s, tj
        if best_s >= 1.0:
            occ[ti] = best_t
    return occ


def judge(frame, t_cap, dets, polys):
    """추적 + IN/OUT 판정. 그리기용 목록을 돌려줌."""
    global total, in_cnt, cur_box
    tracks = tracker.update(Boxes(dets.copy(), frame.shape[:2]), frame)   # [x1,y1,x2,y2,id,score,cls,idx]
    now_t = t_cap
    cur_box = {}
    items = []
    if len(tracks):
        ids = tracks[:, 4].astype(int).tolist()
        boxes = tracks[:, :4]
        scores = tracks[:, 5]
        det_idx = tracks[:, -1].astype(int)
        visible = set(ids)
        cur_box = {t: b for t, b in zip(ids, boxes)}
        empty = np.zeros((0, 2), np.float32)
        tpolys = [polys[di] if 0 <= di < len(polys) else empty for di in det_idx]

        occ_now = occlusion_map(ids, boxes, tpolys)
        for ti in ids:
            if ti in occ_now:
                occluder_of[ti] = occ_now[ti]
            else:
                occluder_of.pop(ti, None)

        for tid, box, sc, poly in zip(ids, boxes, scores, tpolys):
            cx, cy, py, by_mask = judge_point(box, poly)
            prev = zone_of.get(tid)
            new = get_zone(cx, py, prev)
            last_seen[tid] = now_t

            if prev is None:
                best, best_d, how = None, float("inf"), ""
                guard = LINK_GUARD > 0 and new == "B"          # B 에서 새로 생긴 ID: 숨은 B 출신의 것을 가져가지 않음
                for old, occ in list(occluder_of.items()):
                    if old in visible or now_t - last_seen.get(old, -1e9) > REAPPEAR_SEC:
                        continue
                    if guard and origin_of.get(old) == "B" and (LINK_GUARD >= 2 or zone_of.get(old) == "B"):
                        continue
                    anchor = visible_anchor(occ)
                    if anchor is None:
                        continue
                    ox1, oy1, ox2, oy2 = cur_box[anchor]
                    m = (ox2 - ox1) * OCCLUDER_NEAR
                    if not (ox1 - m <= cx <= ox2 + m and oy1 - m <= cy <= oy2 + m):
                        continue
                    lx, ly = last_pos.get(old, (cx, cy))
                    d = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                    if d < best_d:
                        best, best_d, how = old, d, f"id{anchor} 뒤에서 다시 나옴"
                if best is None:
                    best_d = MERGE_DIST
                    for old, (ox, oy) in last_pos.items():
                        if old in visible:
                            continue
                        if guard and origin_of.get(old) == "B" and (LINK_GUARD >= 2 or zone_of.get(old) == "B"):
                            continue
                        d = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                        if d < best_d:
                            best, best_d, how = old, d, "근처에서 끊김"
                if best is not None and best in origin_of:
                    origin_of[tid] = origin_of.pop(best)
                    prev = zone_of.pop(best, None)
                    new = get_zone(cx, py, prev)
                    last_seen.pop(best, None)
                    last_pos.pop(best, None)
                    occluder_of.pop(best, None)
                    if DEBUG:
                        log(f"[연결]  id{best} -> id{tid}  ({how})  {prev}->{new}  출신={origin_of[tid]}")
                else:
                    origin_of[tid] = new
                    in_band = BBAND > 0 and new == "A" and edge_dist(cx, py) <= BBAND
                    if in_band:                                # A 안쪽 띠에서 생긴 ID 는 교실 사람으로 봄
                        origin_of[tid] = "B"
                    if DEBUG:
                        log(f"[새 ID] id{tid}  ({cx:.0f},{py:.0f})  {'띠 안  ' if in_band else ''}출신={origin_of[tid]}")

            if prev is not None and new != prev:
                if prev == "A" and new == "B" and origin_of[tid] == "A":
                    in_cnt += 1
                    total += 1
                    origin_of[tid] = "B"
                    save_total()
                    if DEBUG:
                        log(f"[IN ]   id{tid}  A->B  학생 수={total}")
                elif DEBUG:
                    log(f"[이동]  id{tid}  {prev}->{new}")
            zone_of[tid] = new
            last_pos[tid] = (cx, cy)
            items.append((tid, box, (cx, py, by_mask), f"person: {sc:.2f}  id{tid} {origin_of[tid]}>{new}", poly))

    for tid in list(last_seen):
        gap = now_t - last_seen[tid]
        if tid in occluder_of:
            if gap > REAPPEAR_SEC:
                finalize(tid, f"id{occluder_of.get(tid)} 뒤로 숨은 뒤 안 나타남")
        elif gap > LOST_SEC:
            finalize(tid, "사라짐")
    return items


# ================= 4. 그리기 =================
def draw(frame, items):
    if items:
        overlay = frame.copy()
        for tid, box, pt, label, poly in items:
            if len(poly) >= 3:
                cv2.fillPoly(overlay, [poly.astype(np.int32)], PALETTE[tid % len(PALETTE)])
        cv2.addWeighted(overlay, MASK_ALPHA, frame, 1 - MASK_ALPHA, 0, dst=frame)
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.45 * S
    for tid, (x1, y1, x2, y2), (cx, py, by_mask), label, poly in items:
        col = PALETTE[tid % len(PALETTE)]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, max(1, LW // 2))
        cv2.circle(frame, (int(cx), int(py)), max(3, round(4 * S)), (255, 255, 255), -1 if by_mask else 2)
        if DRAW_LABEL:
            (tw, th), base = cv2.getTextSize(label, font, fs, 1)
            ty = max(th + base + 2, y1)
            cv2.rectangle(frame, (x1, ty - th - base - 2), (x1 + tw + 4, ty), col, -1)
            cv2.putText(frame, label, (x1 + 2, ty - base - 1), font, fs, (255, 255, 255), 1, cv2.LINE_AA)
    col = (0, 0, 255)                                     # A 구역
    top = AT if AT is not None else 0
    cv2.line(frame, (AL, top), (AL, AB - R_BL), col, LW)
    cv2.line(frame, (AR, top), (AR, AB - R_BR), col, LW)
    cv2.line(frame, (AL + R_BL, AB), (AR - R_BR, AB), col, LW)
    if AT is not None:
        cv2.line(frame, (AL, AT), (AR, AT), col, LW)
    if R_BL > 0:
        cv2.ellipse(frame, (AL + R_BL, AB - R_BL), (R_BL, R_BL), 0, 90, 180, col, LW)
    if R_BR > 0:
        cv2.ellipse(frame, (AR - R_BR, AB - R_BR), (R_BR, R_BR), 0, 0, 90, col, LW)
    if DRAW_BAND and BBAND > 0:                           # 띠의 안쪽 경계 (이 선과 빨간 선 사이에서 생긴 ID 는 출신 B)
        col2, r2 = (0, 200, 255), max(0, R_BL - BBAND)
        x2, y2 = AL + BBAND, AB - BBAND
        if y2 - r2 > top and AR > x2 + r2:
            cv2.line(frame, (x2, top), (x2, y2 - r2), col2, 1)
            cv2.line(frame, (x2 + r2, y2), (AR, y2), col2, 1)
            if r2 > 0:
                cv2.ellipse(frame, (x2 + r2, y2 - r2), (r2, r2), 0, 90, 180, col2, 1)
    if DRAW_HUD:
        for i, txt in enumerate([f"Total: {total}", f"IN {in_cnt} / OUT {out_cnt}", f"FPS {fps_now:.1f}"]):
            cv2.putText(frame, txt, (10, int((30 + 30 * i) * S)), cv2.FONT_HERSHEY_SIMPLEX,
                        (0.8 if i == 0 else 0.6) * S, (255, 255, 255), LW)
    return frame


def write_stats():
    if not APP_LINK:
        return
    write_atomic(STATS_PATH, json.dumps({
        "total": total, "in": in_cnt, "out": out_cnt, "fps": round(fps_now, 1),
        "workers": len(workers_up), "ts": time.time()}).encode("utf-8"))


# ================= 5. 판정 루프 (별도 스레드) =================
show_lock = threading.Lock()
show_img = None                   # 화면에 띄울 최신 완성 프레임
show_no = 0
stop_event = threading.Event()


def set_total_now(v, who):
    """인원 설정 (판정 스레드 안에서만 호출). IN/OUT 은 0 으로 초기화."""
    global total, in_cnt, out_cnt, TOTAL_START
    total, in_cnt, out_cnt, TOTAL_START = v, 0, 0, v
    save_total()
    write_stats()
    log(f"[{who}] 현재 인원 설정 -> {total}명 (IN/OUT 0으로 초기화)")


def process_loop():
    global expect, need_reset, ctrl_cmd, fps_now, show_img, show_no
    first_wait = None
    n_proc = n_skip = 0
    t_win, n_win = time.time(), 0
    t_log, t_stats = time.time(), 0.0
    ms_sum = 0.0
    while not stop_event.is_set():
        # ---- 인원 설정 (키 r / 웹 앱. 영상이 안 들어와도 바로 적용) ----
        if ctrl_cmd is not None:
            with ctrl_lock:
                v, ctrl_cmd = ctrl_cmd, None
            set_total_now(v, "설정")
            ctrl_done.set()

        # ---- 다음 번호 프레임 꺼내기 (재정렬) ----
        pkt = None
        with buf_cond:
            if not buffer:
                buf_cond.wait(0.1)
            if need_reset:
                need_reset = False
                reset_tracking()
            if buffer:
                if expect is None:
                    expect = min(buffer)
                for k in [k for k in buffer if k < expect]:     # 이미 지나간 번호는 버림
                    del buffer[k]
                if expect in buffer:
                    pkt = buffer.pop(expect)
                    expect += 1
                    first_wait = None
                elif buffer:
                    now = time.time()
                    if first_wait is None:
                        first_wait = now
                    elif now - first_wait > REORDER_TIMEOUT or len(buffer) > 30:
                        lo = min(buffer)
                        n_skip += lo - expect
                        expect, first_wait = lo, None

        if pkt is not None:
            t0 = time.time()
            t_cap, jpg, dets, polys = pkt
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                if (FRAME_W, FRAME_H) != (frame.shape[1], frame.shape[0]):
                    set_geometry(frame.shape[1], frame.shape[0])
                items = judge(frame, t_cap, dets, polys)
                draw(frame, items)
                if SHOW:
                    with show_lock:
                        show_img, show_no = frame, show_no + 1
                if APP_LINK:
                    ok, out = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, OUT_JPEG_QUALITY])
                    if ok:
                        write_atomic(FRAME_PATH, out.tobytes())
                n_proc += 1
                n_win += 1
                ms_sum += (time.time() - t0) * 1000

        now = time.time()
        if now - t_win >= 1.0:                       # 처리 속도 (1초 창)
            fps_now = n_win / (now - t_win)
            t_win, n_win = now, 0
        if pkt is not None or now - t_stats >= 1.0:  # 웹 앱용 숫자 (영상 없어도 1초마다)
            write_stats()
            t_stats = now
        if now - t_log >= 5:
            ws = "  ".join(f"{ip.split(':')[0].split('.')[-1]}번:{ms:.0f}ms"
                           for ip, ms in sorted(worker_ms.items()) if ip in workers_up) or "없음"
            log(f"[마스커] FPS={fps_now:.1f}  처리 {n_proc}장  건너뜀 {n_skip}"
                f"  판정+그리기 {ms_sum / max(1, n_proc):.0f}ms  작업자[{ws}]"
                f"  학생 수={total} (IN={in_cnt} OUT={out_cnt})")
            n_proc = n_skip = 0
            ms_sum = 0.0
            t_log = now


def request_total(v):
    """다른 스레드(키 입력 등)에서 인원 설정 요청 -> 판정 스레드가 적용."""
    global ctrl_cmd
    with ctrl_lock:
        ctrl_cmd = v


# ================= 6. 화면 출력 (메인 스레드: X11) =================
def display_loop():
    """최신 완성 프레임만 띄움. 화면이 느려도 판정 스레드는 영향 없음."""
    last_no, t_last = -1, 0.0
    gap = 1.0 / max(1, SHOW_MAX_FPS)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    while not stop_event.is_set():
        img = None
        with show_lock:
            if show_no != last_no and time.time() - t_last >= gap:
                img, last_no = show_img, show_no
        if img is not None:
            if SHOW_SCALE != 1.0:
                img = cv2.resize(img, None, fx=SHOW_SCALE, fy=SHOW_SCALE, interpolation=cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, img)
            t_last = time.time()
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            request_total(TOTAL_START)
    cv2.destroyAllWindows()


def main():
    global SHOW
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    if SHOW and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        log("[마스커] DISPLAY 없음 -> X11 화면 출력 끔 (X11 포워딩 SSH 나 모니터에서 실행하면 켜짐)")
        SHOW = False

    threading.Thread(target=accept_loop, daemon=True).start()
    if APP_LINK:
        threading.Thread(target=control_loop, daemon=True).start()
    write_stats()
    th = threading.Thread(target=process_loop, daemon=True)
    th.start()

    if SHOW:
        log(f"[마스커] X11 화면 출력 켜짐 (최대 {SHOW_MAX_FPS}fps, 배율 {SHOW_SCALE})  q: 종료  r: 학생 수 리셋")
        display_loop()
    else:
        log("[마스커] 화면 없이 실행 중 (Ctrl+C 로 종료)")
        while th.is_alive():
            time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        log(f"[마스커] 종료  학생 수={total}  IN={in_cnt}  OUT={out_cnt}")

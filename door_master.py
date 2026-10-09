# door_master.py  (마스터 Pi 10.10.17.33: 카메라가 연결된 Pi)
# ─────────────────────────────────────────────────────────────────────────────
# 4대 구성
#   마스터 : 카메라 -> JPEG 압축 -> 작업자 2대에게 분배                    (이 파일)
#   작업자 : 추론(yolo26s-seg) + 사람 윤곽 좌표 추출 -> 마스커로 직접 전송   (door_worker.py)
#   마스커 : 순서 정렬 + 추적 + IN/OUT 판정 + 마스크 그리기 + X11 출력      (door_masker.py)
#
# 마스터는 추론 결과를 받지 않음. 작업자가 '다 했다(ack)'고 알려주면 그 작업자에게 다음 프레임을 줌.
# 비어 있는 작업자가 가장 최신 프레임을 가져가므로 빠른 작업자가 더 많이 처리함 (자동 부하 분산).
# 마스터에는 ultralytics/torch 가 필요 없음 (opencv, numpy 만).
#
# 통신 형식 (TCP)
#   마스터 -> 작업자 : [프레임번호 4B][실행ID 4B][촬영시각 8B double][JPEG길이 4B][JPEG]
#   작업자 -> 마스터 : [프레임번호 4B][사람 수 4B][추론시간ms 4B float]   (ack)
#
# 실행 순서: 마스커 -> 작업자 2대 -> 마스터
#   source ~/work/yolo-env/bin/activate && python door_master.py
# 종료: 터미널에 q 입력 후 Enter, 또는 Ctrl+C
# ─────────────────────────────────────────────────────────────────────────────

import os

# 수학 라이브러리 스레드 제한 + 바쁜 대기 끄기 (numpy / cv2 import '전에')
for _k, _v in {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
               "OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0", "KMP_BLOCKTIME": "0"}.items():
    os.environ.setdefault(_k, _v)

import random
import socket
import struct
import sys
import threading
import time
import cv2

cv2.setNumThreads(2)

# ---------------- 설정 ----------------
WORKERS = [                       # 작업자 Pi (IP, 포트)
    ("10.10.17.124", 6000),
    ("10.10.17.117", 6000),
]
JPEG_QUALITY = 80                 # 작업자로 보낼 JPEG 품질
CAM_INDEX = 0
CAM_W, CAM_H = 640, 480
ROTATE = None                     # None, "CW", "CCW", "180"

ROT_CODE = {None: None, "CW": cv2.ROTATE_90_CLOCKWISE,
            "CCW": cv2.ROTATE_90_COUNTERCLOCKWISE, "180": cv2.ROTATE_180}
assert ROTATE in ROT_CODE

RUN_ID = random.getrandbits(32)   # 마스터를 다시 켜면 바뀜 -> 마스커가 프레임 번호를 새로 맞춤
HDR_TO_WORKER = struct.Struct("!IIdI")   # seq, run_id, t_cap, jpeg_len
ACK = struct.Struct("!IIf")              # seq, n_people, infer_ms

running = True


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("연결 끊김")
        buf.extend(chunk)
    return bytes(buf)


# ================= 1. 카메라 읽기 스레드 =================
class FrameGrabber:
    """카메라를 계속 읽어 최신 프레임 1장만 보관. 새 프레임이 오면 기다리던 작업자 스레드를 깨움."""

    def __init__(self):
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        assert self.cap.isOpened(), "웹캠을 열 수 없음 (다른 프로그램이 쓰는 중인지, 번호가 맞는지 확인)"
        self.cond = threading.Condition()
        self.frame, self.fid, self.t = None, 0, 0.0
        self.n_read = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if ROT_CODE[ROTATE] is not None:
                f = cv2.rotate(f, ROT_CODE[ROTATE])
            with self.cond:
                self.frame, self.fid, self.t = f, self.fid + 1, time.time()
                self.n_read += 1
                self.cond.notify_all()


grab = FrameGrabber()
while grab.frame is None:
    time.sleep(0.01)
print(f"[마스터] 카메라 {grab.frame.shape[1]}x{grab.frame.shape[0]}  실행ID={RUN_ID:08x}", flush=True)

# ================= 2. 프레임 분배 (작업자마다 스레드 1개) =================
# 비어 있는 작업자가 '가장 최신 프레임'을 가져감 -> 빠른 작업자가 더 많이 처리 (자동 부하 분산)
disp_lock = threading.Lock()
next_seq = 0
last_fid = 0
stat_lock = threading.Lock()
stat = {"sent": 0, "enc": 0.0}
worker_ms = {}                    # 작업자별 최근 추론 시간
worker_people = {}                # 작업자별 최근 사람 수
worker_ok = {}


def take_frame():
    """카메라의 새 프레임 하나에 순번을 붙여 가져감. 같은 프레임을 두 작업자에게 주지 않음."""
    global next_seq, last_fid
    while running:
        with grab.cond:
            grab.cond.wait_for(lambda: grab.fid != last_fid or not running, timeout=0.5)
            fid, f, t = grab.fid, grab.frame, grab.t
        with disp_lock:
            if fid != last_fid:
                last_fid = fid
                seq = next_seq
                next_seq += 1
                return seq, f, t
    return None, None, None


def worker_loop(wid, host, port):
    sock = None
    while running:
        if sock is None:
            try:
                sock = socket.create_connection((host, port), timeout=3)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(5)
                worker_ok[wid] = True
                print(f"[마스터] 작업자{wid} 연결됨 {host}:{port}", flush=True)
            except OSError:
                worker_ok[wid] = False
                time.sleep(2)
                continue
        seq, f, t_cap = take_frame()
        if seq is None:
            break
        t0 = time.time()
        ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        jb = jpg.tobytes()
        enc = time.time() - t0
        try:
            sock.sendall(HDR_TO_WORKER.pack(seq, RUN_ID, t_cap, len(jb)) + jb)
            _, n, ms = ACK.unpack(recv_exact(sock, ACK.size))     # 작업자가 마스커로 보낸 뒤 ack
            worker_ms[wid], worker_people[wid] = ms, n
            with stat_lock:
                stat["sent"] += 1
                stat["enc"] += enc
        except (OSError, ConnectionError):
            print(f"[마스터] 작업자{wid} 연결 끊김, 다시 시도 (그 프레임은 마스커에서 건너뜀)", flush=True)
            worker_ok[wid] = False
            sock.close()
            sock = None


for i, (h, p) in enumerate(WORKERS):
    threading.Thread(target=worker_loop, args=(i, h, p), daemon=True).start()


# ================= 3. 터미널 명령 =================
def key_loop():
    global running
    try:
        for line in sys.stdin:
            if line.strip().lower() == "q":
                running = False
                break
    except (OSError, ValueError):
        pass


threading.Thread(target=key_loop, daemon=True).start()

# ================= 4. 상태 출력 =================
t_stat, cam_last = time.time(), 0
try:
    while running:
        time.sleep(0.2)
        now = time.time()
        if now - t_stat < 3:
            continue
        dt = now - t_stat
        with stat_lock:
            sent, enc = stat["sent"], stat["enc"]
            stat.update(sent=0, enc=0.0)
        cam_fps = (grab.n_read - cam_last) / dt
        cam_last = grab.n_read
        ws = "  ".join(
            (f"w{k}:{worker_ms.get(k, 0):.0f}ms/{worker_people.get(k, 0)}명" if worker_ok.get(k) else f"w{k}:끊김")
            for k in range(len(WORKERS)))
        print(f"[마스터] 카메라 {cam_fps:4.1f}fps  분배 {sent / dt:4.1f}장/초"
              f"  인코딩 {1000 * enc / max(1, sent):.1f}ms  작업자[{ws}]", flush=True)
        t_stat = now
except KeyboardInterrupt:
    pass
finally:
    running = False
    print("[마스터] 종료", flush=True)

# door_worker.py  (작업자 Pi 2대 10.10.17.124 / .117: 추론 + 사람 윤곽 좌표 추출)
# ─────────────────────────────────────────────────────────────────────────────
# 마스터(door_master.py)가 보낸 JPEG 를 받아 yolo26s-seg 로 사람을 검출하고
# 사람마다 박스 + 윤곽(폴리곤) 좌표를 뽑아 '마스커(door_masker.py)로 직접' 보냄.
# 마스터에게는 "다 했다(ack)" 만 짧게 돌려줌 -> 마스터가 다음 프레임을 줌.
# 마스커가 꺼져 있어도 추론은 계속함 (2초마다 다시 연결, 그동안 결과는 버림).
#
# 모델: yolo26s-seg (ncnn, fp16). 처음 한 번 아래 명령으로 변환해서 models/ 에 넣어야 함.
#   yolo export model=yolo26s-seg.pt format=ncnn half=True imgsz=320
#   mv yolo26s-seg_ncnn_model models/yolo26s_seg_fp16_ncnn_model
#
# 통신 형식 (TCP)
#   마스터 -> 작업자 : [프레임번호 4B][실행ID 4B][촬영시각 8B double][JPEG길이 4B][JPEG]
#   작업자 -> 마스터 : [프레임번호 4B][사람 수 4B][추론시간ms 4B float]   (ack)
#   작업자 -> 마스커 : [프레임번호 4B][실행ID 4B][촬영시각 8B][추론시간ms 4B float][사람 수 N 4B][JPEG길이 4B]
#                      [JPEG][N x 6 float32 (x1,y1,x2,y2,conf,cls)][N x 점개수 uint32][모든 윤곽 점 (x,y) float32]
#
# 실행: source ~/work/yolo-env/bin/activate && python door_worker.py
# ─────────────────────────────────────────────────────────────────────────────

import os

# ---- OpenMP 바쁜 대기(busy-wait) 끄기: 반드시 torch / ultralytics(ncnn) import '전에' ----
PASSIVE_WAIT = True
if PASSIVE_WAIT:
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("GOMP_SPINCOUNT", "0")
    os.environ.setdefault("KMP_BLOCKTIME", "0")

import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---------------- 설정 (모든 작업자 Pi에서 같게) ----------------
BASE = Path(__file__).resolve().parent
MODEL = BASE / "models" / "yolo26s_seg_fp16_ncnn_model"
IMGSZ = 320                  # export 할 때의 imgsz 와 반드시 같게
CONF = 0.20
IOU = 0.60
PORT = 6000                  # 마스터가 접속하는 포트
MASKER = ("10.10.17.105", 6100)   # 결과를 보낼 마스커 Pi
SPIKE_MS = 120               # 추론이 이보다 오래 걸리면 그 순간의 CPU 상태 출력 (seg 는 평소 60ms 대)
TORCH_THREADS = 1            # 후처리(PyTorch) 스레드 수. NCNN 과 코어를 다투지 않게 1로 제한
SAVE_SAMPLE_EVERY = 200      # 이 장마다 받은 프레임을 sample_frame.jpg 로 저장. 0이면 안 함
POLY_EPS = 1.5               # 윤곽 단순화 정도(픽셀). 클수록 점이 줄어 전송량 감소

if TORCH_THREADS:
    torch.set_num_threads(TORCH_THREADS)

HDR_FROM_MASTER = struct.Struct("!IIdI")    # seq, run_id, t_cap, jpeg_len
ACK = struct.Struct("!IIf")                 # seq, n_people, infer_ms
HDR_TO_MASKER = struct.Struct("!IIdfII")    # seq, run_id, t_cap, infer_ms, n, jpeg_len


def cpu_state(with_throttle=False):
    """CPU 클럭, 온도, 부하."""
    try:
        freq = int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read()) // 1000
    except OSError:
        freq = -1
    try:
        temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except OSError:
        temp = -1
    thr = ""
    if with_throttle:
        try:
            thr = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1).stdout.strip()
        except Exception:
            thr = "?"
    return f"클럭 {freq}MHz  온도 {temp:.1f}C  {thr}  부하(1분) {os.getloadavg()[0]:.2f}"


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("연결 끊김")
        buf.extend(chunk)
    return bytes(buf)


def get_polys(r, n):
    """검출 n개 각각의 윤곽 폴리곤 (원본 이미지 좌표). 박스 순서와 같음."""
    empty = np.zeros((0, 2), np.float32)
    if r.masks is None:
        return [empty] * n
    polys = []
    for p in r.masks.xy:
        p = np.asarray(p, np.float32)
        if len(p) >= 3:
            p = cv2.approxPolyDP(p.reshape(-1, 1, 2), POLY_EPS, True).reshape(-1, 2).astype(np.float32)
        else:
            p = empty
        polys.append(p)
    polys += [empty] * (n - len(polys))
    return polys[:n]


class MaskerLink:
    """마스커 Pi 연결. 끊겨도 추론은 멈추지 않음 (2초마다 다시 연결 시도, 그동안 결과는 버림)."""

    def __init__(self, addr):
        self.addr, self.sock, self.next_try = addr, None, 0.0

    def send(self, data):
        if self.sock is None:
            if time.time() < self.next_try:
                return False
            try:
                self.sock = socket.create_connection(self.addr, timeout=1)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.settimeout(2)
                print(f"[작업자] 마스커 연결됨 {self.addr[0]}:{self.addr[1]}", flush=True)
            except OSError:
                self.sock, self.next_try = None, time.time() + 2
                return False
        try:
            self.sock.sendall(data)
            return True
        except OSError:
            print("[작업자] 마스커 연결 끊김, 다시 시도", flush=True)
            self.sock.close()
            self.sock, self.next_try = None, time.time() + 2
            return False


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    if not MODEL.exists():
        sys.exit(f"[작업자] 모델 폴더가 없음: {MODEL}\n"
                 f"  -> yolo export model=yolo26s-seg.pt format=ncnn half=True imgsz={IMGSZ}\n"
                 f"  -> mv yolo26s-seg_ncnn_model {MODEL}")

    model = YOLO(str(MODEL), task="segment")
    dummy = np.zeros((480, 640, 3), np.uint8)
    for _ in range(3):                                   # 워밍업
        model.predict(dummy, imgsz=IMGSZ, verbose=False)
    print(f"[작업자] 모델 준비 완료: {MODEL.name}")
    print(f"[작업자] 시작 상태: {cpu_state(True)}  torch 스레드={torch.get_num_threads()}")

    masker = MaskerLink(MASKER)
    masker.send(b"")                                     # 미리 연결 시도

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(1)
    print(f"[작업자] {PORT} 포트에서 마스터 접속 대기 중...")

    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[작업자] 마스터 연결됨: {addr[0]}")
        n_done, t_stat, ms_sum, ms_max, n_spike, box_sum, n_fail = 0, time.time(), 0.0, 0.0, 0, 0, 0
        try:
            while True:
                seq, run_id, t_cap, jlen = HDR_FROM_MASTER.unpack(recv_exact(conn, HDR_FROM_MASTER.size))
                jpg = recv_exact(conn, jlen)
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if SAVE_SAMPLE_EVERY and seq % SAVE_SAMPLE_EVERY == 0:
                    cv2.imwrite(str(BASE / "sample_frame.jpg"), img)

                # ---- 추론 + 윤곽 좌표 추출 ----
                t0 = time.time()
                r = model.predict(img, imgsz=IMGSZ, conf=CONF, iou=IOU, classes=[0], verbose=False)[0]
                b = r.boxes
                n = len(b)
                if n:
                    arr = np.hstack([b.xyxy.cpu().numpy(),
                                     b.conf.cpu().numpy()[:, None],
                                     b.cls.cpu().numpy()[:, None]]).astype(np.float32)
                else:
                    arr = np.zeros((0, 6), np.float32)
                polys = get_polys(r, n)
                ms = (time.time() - t0) * 1000                  # 윤곽 추출까지 포함

                if ms > SPIKE_MS:
                    n_spike += 1
                    print(f"[튐] {time.strftime('%H:%M:%S')}  #{seq}  추론 {ms:.0f}ms  박스 {n}개  |  {cpu_state()}")

                # ---- 마스커로 전송 (JPEG 는 받은 것 그대로 재사용) ----
                counts = np.array([len(p) for p in polys], dtype=">u4")
                pts = np.concatenate(polys).astype(np.float32) if n else np.zeros((0, 2), np.float32)
                pkt = (HDR_TO_MASKER.pack(seq, run_id, t_cap, ms, n, len(jpg)) + jpg
                       + arr.tobytes() + counts.tobytes() + pts.tobytes())
                if not masker.send(pkt):
                    n_fail += 1

                # ---- 마스터에 ack ----
                conn.sendall(ACK.pack(seq, n, ms))

                n_done += 1
                ms_sum += ms
                ms_max = max(ms_max, ms)
                box_sum += n
                if time.time() - t_stat >= 5:
                    rs = "마스커 연결됨" if masker.sock else "마스커 끊김"
                    print(f"[작업자] {time.strftime('%H:%M:%S')}  {n_done / (time.time() - t_stat):.1f}장/초"
                          f"  평균 추론 {ms_sum / n_done:.0f}ms  최대 {ms_max:.0f}ms  튐 {n_spike}회"
                          f"  평균 사람 {box_sum / n_done:.1f}명  {rs}(전송 실패 {n_fail})")
                    n_done, t_stat, ms_sum, ms_max, n_spike, box_sum, n_fail = 0, time.time(), 0.0, 0.0, 0, 0, 0
        except (ConnectionError, OSError) as e:
            print(f"[작업자] 마스터 연결 끊김 ({e}), 다시 대기")
            conn.close()


if __name__ == "__main__":
    main()


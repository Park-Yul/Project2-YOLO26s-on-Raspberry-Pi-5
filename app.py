#!/usr/bin/env python3
"""
라즈베리파이5 door_counter 웹 컨트롤러 (디자인 리뉴얼 버전)
- 로그인(SSH 인증) -> 메인(초기 인원수 설정 / 실행)
- 실행 화면: 웹캠 영상 + 현재 인원 / IN / OUT 실시간 표시
- 이 파일은 라즈베리파이(10.10.17.105) 위에서 실행해야 합니다.

설치:  pip install flask paramiko
실행:  python3 app.py        ->  http://10.10.17.105:8000
"""
import atexit
import json
import os
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time

import paramiko
from flask import (Flask, Response, jsonify, redirect, render_template_string,
                   request, session, url_for)

PI_HOST = os.environ.get("PI_HOST", "10.10.17.105")
PI_PORT = int(os.environ.get("PI_PORT", "22"))
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))

# Busy Wait 설정
# True  : cv2.waitKey() 대기 시간 동안 CPU를 계속 사용하며 대기
# False : time.sleep()으로 CPU를 양보하며 대기 (일반적으로 권장)
BUSY_WAIT = True

# 마스커 실행 전에 넣는 스레드 설정 (환경 변수)
# 래퍼(stream_runner.py)가 cv2 를 먼저 import 하므로, 마스커 파일 안의 설정은 이미 늦음.
# 파이썬이 시작되기 전에 넣어야 수학 라이브러리가 스레드 1개 + 기다릴 때 잠들기로 올라옴.
THREAD_ENV = ("OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE "
              "GOMP_SPINCOUNT=0 KMP_BLOCKTIME=0")

VENV_ACTIVATE = "work/yolo-env/bin/activate"
SCRIPT_NAME = "masker.py"      # 홈디렉터리 아래에서 자동 검색

# masker.py 웹 제어 포트
CONTROL_HOST = "127.0.0.1"
CONTROL_PORT = 6200

# TOTAL_START = 숫자 형태의 줄
# door_total.json이 없을 때 사용할 fallback 값도 같이 맞춰두기 위해 사용
TOTAL_RE = re.compile(r"^(\s*TOTAL_START\s*=(?!=)\s*)(-?\d+)(?!\d)")
FRAME_PATH = "/tmp/door_frame.jpg"
STATS_PATH = "/tmp/door_stats.json"
RUNNER_PATH = "/tmp/stream_runner.py"
LOG_PATH = "/tmp/door_counter.log"
PID_PATH = "/tmp/door_counter.pid"

# 웹페이지 닫힘 감지(heartbeat) 설정
LEAVE_GRACE = 10     # 탭 닫힘 신호(/leaving) 후, 이 시간(초) 동안 heartbeat가 없으면 종료
HB_TIMEOUT = 90      # 신호 없이 heartbeat가 이 시간(초) 끊기면 종료 (브라우저 강제종료 등)

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
SESSIONS = {}  # token -> {"client": SSHClient, "user": str, "venv": str}


# ---------------------------------------------------------------------------
# door_counter.py 를 그대로 실행하되, cv2.imshow 화면을 JPEG 파일로 내보내는 래퍼
# (SSH/서버에는 모니터가 없으므로 창 대신 웹으로 화면을 전달)
# + 화면에 그려지는 "Total: N (IN a / OUT b)" 문구를 가로채 숫자를 JSON으로 저장
#   (실행 화면의 현재 인원 / IN / OUT 표시에 사용. door_counter.py 수정 불필요)
# ---------------------------------------------------------------------------
RUNNER_CODE = r'''
import os, sys, time, runpy, re, json
import cv2

# app.py에서 DOOR_BUSY_WAIT 환경변수로 전달받음
BUSY_WAIT = os.environ.get("DOOR_BUSY_WAIT", "0") == "1"

FRAME = "/tmp/door_frame.jpg"
STATS = "/tmp/door_stats.json"

# door_counter.py는 통계를 두 줄로 따로 그림:
#   Total: 10
#   IN 2 / OUT 1
# 이 두 문자열을 가로채 웹 카드용 JSON에 저장하고,
# 카메라 영상에는 Total / IN / OUT 문구를 표시하지 않는다.
# 카메라 HUD에서 TOTAL / IN / OUT / FPS를 찾아서
# 값은 웹 대시보드용으로 저장하고, 카메라 영상에는 그리지 않는다.
_pat_total = re.compile(r"\bTOTAL\s*[:=]?\s*(-?\d+)", re.I)
_pat_in    = re.compile(r"\bIN\s*[:=]?\s*(\d+)", re.I)
_pat_out   = re.compile(r"\bOUT\s*[:=]?\s*(\d+)", re.I)
_pat_fps   = re.compile(r"\bFPS\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", re.I)

_orig_putText = cv2.putText
_orig_line = cv2.line
_orig_ellipse = cv2.ellipse

_last_total = None
_last_in = None
_last_out = None

def _save_stats():
    """현재까지 잡힌 TOTAL / IN / OUT 값을 웹 대시보드용 JSON으로 저장."""
    if _last_total is None or _last_in is None or _last_out is None:
        return

    with open(STATS + ".tmp", "w") as f:
        json.dump({
            "total": _last_total,
            "in": _last_in,
            "out": _last_out,
            "ts": time.time()
        }, f)

    os.replace(STATS + ".tmp", STATS)

def _putText(img, text, *a, **k):
    global _last_total, _last_in, _last_out

    try:
        s = str(text)

        # TOTAL 값 읽기
        m = _pat_total.search(s)
        if m:
            _last_total = int(m.group(1))

        # IN 값 읽기
        m = _pat_in.search(s)
        if m:
            _last_in = int(m.group(1))

        # OUT 값 읽기
        m = _pat_out.search(s)
        if m:
            _last_out = int(m.group(1))

        _save_stats()

        # 카메라 영상 안의 TOTAL / IN / OUT / FPS 글자만 숨김
        if (_pat_total.search(s) or
            _pat_in.search(s) or
            _pat_out.search(s) or
            _pat_fps.search(s)):
            return img

        # A / B 구역 글자도 기존처럼 숨김
        if s.strip() in ("A", "B"):
            return img

    except Exception:
        pass

    # 사람 ID / confidence 등의 글자는 그대로 표시
    return _orig_putText(img, text, *a, **k)

def _is_zone_line(color):
    """구역 경계선(빨강/노랑)은 화면에 표시하지 않음."""
    try:
        b, g, r = (int(v) for v in color[:3])

        # 기존 빨간 구역선 숨김 유지
        if (b, g, r) == (0, 0, 255):
            return True

        # 노란색 계열 구역선 숨김
        # OpenCV 기본 노랑 (0,255,255)뿐 아니라 압축/밝기 차이도 대응
        if b <= 80 and g >= 150 and r >= 150:
            return True

        return False
    except Exception:
        return False

def _line(img, pt1, pt2, color, *a, **k):
    # 빨강/노랑 구역 직선은 화면에만 그리지 않음
    if _is_zone_line(color):
        return img
    return _orig_line(img, pt1, pt2, color, *a, **k)

def _ellipse(img, center, axes, angle, startAngle, endAngle, color, *a, **k):
    # 빨강/노랑 구역 곡선도 화면에만 그리지 않음
    if _is_zone_line(color):
        return img
    return _orig_ellipse(img, center, axes, angle, startAngle, endAngle, color, *a, **k)

def _imshow(name, frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with open(FRAME + ".tmp", "wb") as f:
            f.write(buf.tobytes())
        os.replace(FRAME + ".tmp", FRAME)

def _wait(delay=1):
    # cv2.waitKey(delay)의 delay는 ms 단위
    wait_sec = max(1, int(delay or 1)) / 1000.0

    if BUSY_WAIT:
        # Busy Wait ON: 지정 시간 동안 CPU를 계속 사용하며 대기
        end = time.perf_counter() + wait_sec
        while time.perf_counter() < end:
            pass
    else:
        # Busy Wait OFF: sleep으로 CPU 사용량을 줄임
        time.sleep(wait_sec)

    return -1

def _noop(*a, **k):
    return None

cv2.putText = _putText
cv2.line = _line
cv2.ellipse = _ellipse
cv2.imshow = _imshow
cv2.waitKey = _wait
for _n in ("namedWindow", "destroyAllWindows", "destroyWindow",
           "resizeWindow", "moveWindow", "setWindowProperty"):
    setattr(cv2, _n, _noop)

script = os.path.abspath(sys.argv[1])
sys.argv = [script]
runpy.run_path(script, run_name="__main__")
'''

# ---------------------------------------------------------------------------
# 템플릿
# ---------------------------------------------------------------------------
BASE = """
<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>6조 출입국</title>
<style>
:root{
  --bg:#0a0f1c; --panel:#111a2e; --panel2:#182440; --line:#25324f;
  --fg:#e8edf7; --dim:#8b98b5; --acc:#4cc9f0; --acc-ink:#04202b;
  --in:#5eead4; --out:#ff8a65; --err:#ff6b6b; --ok:#4ade80; --warn:#fbbf24;
}
*{box-sizing:border-box}
html{color-scheme:dark}
body{
  margin:0; min-height:100vh; color:var(--fg); background-color:var(--bg);
  background-image:
    linear-gradient(rgba(76,201,240,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(76,201,240,.045) 1px, transparent 1px);
  background-size:44px 44px;
  font-family:"Pretendard","Noto Sans KR","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
  letter-spacing:-.01em; line-height:1.5;
}
.wrap{max-width:760px; margin:0 auto; padding:28px 24px 56px}
.wrap.wide{max-width:1360px; padding-left:18px; padding-right:18px}
:focus-visible{outline:2px solid var(--acc); outline-offset:3px}

/* 상단 바 */
.top{display:flex; align-items:center; justify-content:space-between; margin-bottom:36px}
.brand{display:flex; align-items:center; gap:10px; font-weight:800; font-size:1.05rem; color:var(--fg); text-decoration:none}
.brand svg{width:26px; height:26px; color:var(--acc)}
.who{display:flex; align-items:center; gap:14px; font-size:.9rem; color:var(--dim)}
.who b{color:var(--fg); font-weight:600}
.who form{margin:0}
.pill{font:inherit; font-size:.9rem; cursor:pointer; color:var(--fg); background:transparent; padding:7px 14px; border:1px solid var(--line); border-radius:999px; transition:border-color .15s, background .15s}
.pill:hover{border-color:var(--acc); background:rgba(76,201,240,.08)}

h1{font-size:2rem; line-height:1.2; margin:0 0 8px; font-weight:800; letter-spacing:-.03em}
.sub{color:var(--dim); margin:0 0 28px; font-size:.95rem; word-break:break-all}

/* 패널 / 폼 */
.panel{background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:28px}
label{display:block; margin:0 0 8px; font-size:.88rem; color:var(--dim); font-weight:600}
.field+.field{margin-top:18px}
input{
  width:100%; padding:14px 16px; border-radius:10px; border:1px solid var(--line);
  background:var(--bg); color:var(--fg); font:inherit; font-size:1rem; transition:border-color .15s, box-shadow .15s;
}
input:focus{outline:none; border-color:var(--acc); box-shadow:0 0 0 3px rgba(76,201,240,.18)}

/* 버튼 */
.btn,button.btn{
  display:inline-flex; align-items:center; justify-content:center; gap:8px; cursor:pointer;
  padding:13px 22px; border-radius:10px; border:1px solid transparent; font:inherit; font-weight:700; font-size:1rem;
  text-decoration:none; color:var(--acc-ink); background:var(--acc); transition:transform .12s, filter .15s, background .15s, border-color .15s;
}
.btn:hover{filter:brightness(1.08)} .btn:active{transform:translateY(1px) scale(.99)}
.btn.ghost{background:transparent; color:var(--fg); border-color:var(--line)}
.btn.ghost:hover{border-color:var(--dim); background:rgba(255,255,255,.04); filter:none}
.btn.danger{background:rgba(255,107,107,.12); color:var(--err); border-color:rgba(255,107,107,.45)}
.btn.danger:hover{background:rgba(255,107,107,.2); filter:none}
.btn.block{width:100%}
.actions{display:flex; gap:10px; margin-top:24px}

/* 메시지 */
.msg{margin:0 0 18px; padding:11px 14px; border-radius:10px; font-size:.9rem; white-space:pre-wrap; background:var(--panel2); border:1px solid var(--line)}
.msg.err{color:var(--err); border-color:rgba(255,107,107,.4); background:rgba(255,107,107,.08)}
.msg.ok{color:var(--ok); border-color:rgba(74,222,128,.35); background:rgba(74,222,128,.08)}
code{font-family:ui-monospace,"JetBrains Mono",Consolas,monospace; color:var(--acc); word-break:break-all}

/* 로그인 */
.center{min-height:calc(100vh - 120px); display:flex; align-items:center; justify-content:center}
.login{width:100%; max-width:420px}
.login .logo{width:52px; height:52px; color:var(--acc); margin-bottom:18px}

/* 메인 */
.chips{display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px}
.status{display:inline-flex; align-items:center; gap:8px; font-size:.88rem; padding:7px 14px; border-radius:999px; border:1px solid var(--line); color:var(--dim)}
.status i{width:8px; height:8px; border-radius:50%; background:#64748b}
.status.ok i{background:var(--ok)} .status.warn i{background:var(--warn)}
.status.ok{color:var(--fg)}
.path{margin-bottom:26px; font-size:.85rem; color:var(--dim)}
.tiles{display:grid; grid-template-columns:1fr 1.25fr; gap:18px}
.tile{
  position:relative; display:flex; flex-direction:column; justify-content:space-between; min-height:250px; padding:28px;
  border-radius:16px; text-decoration:none; overflow:hidden; transition:transform .18s, border-color .18s, box-shadow .18s;
}
.tile:hover{transform:translateY(-3px)}
.tile svg{width:44px; height:44px}
.tile strong{display:block; font-size:1.55rem; letter-spacing:-.03em; line-height:1.25}
.tile span{display:block; margin-top:6px; font-size:.9rem; opacity:.75}
.tile.set{background:var(--panel); border:1px solid var(--line); color:var(--fg)}
.tile.set svg{color:var(--acc)}
.tile.set:hover{border-color:var(--acc); box-shadow:0 10px 30px rgba(76,201,240,.12)}
.tile.go{background:var(--acc); color:var(--acc-ink); border:1px solid var(--acc)}
.tile.go:hover{box-shadow:0 12px 34px rgba(76,201,240,.35)}

/* 인원수 설정 */
.stepper{display:flex; align-items:stretch; gap:10px}
.stepper button{width:64px; border-radius:12px; border:1px solid var(--line); background:var(--panel2); color:var(--fg); font-size:1.8rem; cursor:pointer; transition:border-color .15s, background .15s}
.stepper button:hover{border-color:var(--acc); background:rgba(76,201,240,.1)}
.stepper input{text-align:center; font-size:2.6rem; font-weight:800; padding:14px 0; -moz-appearance:textfield}
.stepper input::-webkit-outer-spin-button,.stepper input::-webkit-inner-spin-button{-webkit-appearance:none; margin:0}
.current{display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 16px; border-radius:10px; background:var(--bg); border:1px dashed var(--line); margin-bottom:22px; font-size:.9rem; color:var(--dim)}

/* 실행 화면 */
.stage{
  display:flex;
  flex-direction:column;
  gap:18px;
}

/* 데스크톱: 카메라를 더 크게, 우측 상태 패널은 컴팩트하게 */
.monitor-layout{
  display:grid;
  grid-template-columns:minmax(0,1fr) 255px;
  gap:14px;
  align-items:stretch;
}

.viewport{
  position:relative;
  width:100%;
  aspect-ratio:4/3;
  background:#000;
  border-radius:16px;
  overflow:hidden;
  border:1px solid var(--line);
  box-shadow:0 16px 40px rgba(0,0,0,.16);
}

.viewport img{
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  object-fit:contain;
  opacity:0;
  transition:opacity .3s;
}

.viewport img.on{opacity:1}

.wait{
  position:absolute;
  inset:0;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  gap:14px;
  color:var(--dim);
  font-size:.95rem;
}

.spin{
  width:30px;
  height:30px;
  border-radius:50%;
  border:3px solid var(--line);
  border-top-color:var(--acc);
  animation:spin 1s linear infinite;
}

@keyframes spin{to{transform:rotate(360deg)}}

.br{
  position:absolute;
  width:26px;
  height:26px;
  border:3px solid var(--acc);
  pointer-events:none;
  z-index:2;
}

.br.tl{top:10px; left:10px; border-right:0; border-bottom:0}
.br.tr{top:10px; right:10px; border-left:0; border-bottom:0}
.br.bl{bottom:10px; left:10px; border-right:0; border-top:0}
.br.brr{bottom:10px; right:10px; border-left:0; border-top:0}

.tag{
  position:absolute;
  top:18px;
  left:48px;
  z-index:3;
  display:flex;
  align-items:center;
  gap:8px;
  padding:6px 13px;
  border-radius:999px;
  background:rgba(10,15,28,.82);
  border:1px solid rgba(255,255,255,.08);
  font-size:.8rem;
  font-weight:700;
  backdrop-filter:blur(8px);
}

.tag i{width:9px; height:9px; border-radius:50%; background:var(--warn)}
.tag.live i{background:var(--err); animation:blink 1.4s ease-in-out infinite}
.tag.off i{background:var(--dim)}
@keyframes blink{50%{opacity:.25}}

/* 우측 실시간 현황 패널 */
.stats-panel{
  min-width:0;
  display:grid;
  grid-template-rows:1.45fr 1fr .72fr;
  gap:14px;
}

.metric-card{
  min-width:0;
  min-height:0;
  display:flex;
  flex-direction:column;
  justify-content:space-between;
  padding:22px 24px;
  border-radius:16px;
  border:1px solid var(--line);
  background:linear-gradient(145deg,var(--panel),var(--panel2));
  overflow:hidden;
}

.metric-card .metric-label{
  color:var(--dim);
  font-size:.95rem;
  font-weight:700;
}

.metric-card .metric-value{
  display:flex;
  align-items:flex-end;
  gap:7px;
  margin-top:16px;
  font-size:3.4rem;
  line-height:.95;
  font-weight:850;
  letter-spacing:-.06em;
  font-variant-numeric:tabular-nums;
}

.metric-card .metric-unit{
  padding-bottom:5px;
  color:var(--dim);
  font-size:.9rem;
  font-weight:700;
  letter-spacing:0;
}

/* 현재 인원은 우측 패널에서 가장 크게 강조 */
.metric-card.total-card{
  position:relative;
  padding:26px;
  border-color:rgba(76,201,240,.46);
  background:
    radial-gradient(circle at 88% 10%,rgba(76,201,240,.22),transparent 34%),
    linear-gradient(145deg,#111d33,var(--panel2));
  box-shadow:0 12px 34px rgba(76,201,240,.08);
}

.metric-card.total-card::before{
  content:"";
  position:absolute;
  top:0;
  left:0;
  right:0;
  height:3px;
  background:linear-gradient(90deg,var(--acc),rgba(76,201,240,.16));
}

.total-card .metric-label{
  color:var(--fg);
  font-size:1rem;
}

.total-card .metric-value{
  font-size:5.4rem;
  color:var(--acc);
  text-shadow:0 0 24px rgba(76,201,240,.14);
}

.total-card .metric-unit{
  padding-bottom:9px;
  font-size:1rem;
}

/* IN / OUT은 나란히 */
.io-grid{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:10px;
  min-height:0;
}

.io-grid .metric-card{
  padding:18px 15px;
}

.io-grid .metric-card .metric-value{
  font-size:2.8rem;
}

.metric-card.in-card{
  border-color:rgba(94,234,212,.25);
}

.metric-card.out-card{
  border-color:rgba(255,138,101,.25);
}

.metric-card.fps-card{
  border-color:rgba(251,191,36,.22);
  flex-direction:row;
  align-items:center;
  justify-content:space-between;
  gap:16px;
}

.fps-card .metric-value{
  margin-top:0;
  font-size:2.45rem;
  white-space:nowrap;
}

.fps-card .metric-unit{
  padding-bottom:3px;
}

.in-card .metric-value{color:var(--in)}
.out-card .metric-value{color:var(--out)}
.fps-card .metric-value{color:var(--fg)}

.pop{animation:pop .35s ease-out}
@keyframes pop{0%{transform:scale(1.10)} 100%{transform:scale(1)}}

.run-actions{
  display:flex;
  flex-wrap:wrap;
  gap:10px;
  margin-top:2px;
}

.run-actions form{margin:0}
.run-actions .btn{min-width:130px}

/* 실행 화면 하단 로그: 기본은 숨김, 로그 보기 버튼을 누르면 표시 */
.inline-log-panel{
  display:none;
  border:1px solid var(--line);
  border-radius:16px;
  background:linear-gradient(145deg,var(--panel),rgba(24,36,64,.72));
  overflow:hidden;
}
.inline-log-panel.open{display:block}
.inline-log-head{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:14px;
  padding:16px 20px;
  border-bottom:1px solid var(--line);
}
.inline-log-head strong{
  font-size:1rem;
  color:var(--fg);
}
.inline-log-head span{
  color:var(--dim);
  font-size:.82rem;
}
pre#inline-logbox{
  margin:0;
  padding:16px 18px;
  max-height:260px;
  overflow:auto;
  background:#060a14;
  color:#b8f5c8;
  font:.82rem/1.55 ui-monospace,"JetBrains Mono",Consolas,monospace;
  white-space:pre-wrap;
  word-break:break-word;
}

@media (max-width:980px){
  .tiles{grid-template-columns:1fr}
  .monitor-layout{grid-template-columns:1fr}
  .stats-panel{
    grid-template-columns:1fr 1fr;
    grid-template-rows:auto auto auto;
  }
  .total-card{grid-column:1 / -1; min-height:180px}
  .io-grid{grid-column:1 / -1}
  .fps-card{grid-column:1 / -1; min-height:110px}
}
@media (max-width:620px){
  .stats-panel{grid-template-columns:1fr}
  .total-card,.io-grid,.fps-card{grid-column:auto}
  .io-grid{grid-template-columns:1fr}
  .metric-card{min-height:140px}
  .total-card{min-height:175px}
  .total-card .metric-value{font-size:4.6rem}
  .metric-card.fps-card{
    flex-direction:column;
    align-items:flex-start;
  }
}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
</style></head><body><div class="wrap {{ 'wide' if wide else '' }}">
{% if user %}
<div class="top">
  <a class="brand" href="{{ url_for('main') }}">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 21V4a1 1 0 0 1 1-1h9a1 1 0 0 1 1 1v17"/><path d="M3 21h18"/><circle cx="12.5" cy="12.5" r=".8" fill="currentColor"/><path d="M19 8l2 2-2 2"/></svg>
    6조 출입국
  </a>
  <div class="who">
    <form method="post" action="{{ url_for('logout') }}"><button class="pill" type="submit">로그아웃</button></form>
  </div>
</div>
{% endif %}
{{ body|safe }}
</div>
{% if hb %}<script>
(function(){
  function ping(){ fetch('/ping',{method:'POST',cache:'no-store',keepalive:true}).catch(function(){}); }
  ping(); setInterval(ping,3000);
  window.addEventListener('pagehide',function(){ try{navigator.sendBeacon('/leaving');}catch(e){} });
})();
</script>{% endif %}
</body></html>
"""

LOGIN = """
<div class="center"><div class="login">
  <svg class="logo" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 21V4a1 1 0 0 1 1-1h9a1 1 0 0 1 1 1v17"/><path d="M3 21h18"/><circle cx="12.5" cy="12.5" r=".8" fill="currentColor"/><path d="M19 8l2 2-2 2"/></svg>
  <h1>6조 출입국</h1>
  <p class="sub">시스템에 접속하려면 로그인해주세요</p>
  <form class="panel" method="post">
    {% if error %}<div class="msg err">{{ error }}</div>{% endif %}
    <div class="field"><label for="u">Username</label><input id="u" name="username" autocomplete="username" required autofocus></div>
    <div class="field"><label for="p">Password</label><input id="p" name="password" type="password" autocomplete="current-password" required></div>
    <div class="actions"><button class="btn block" type="submit">접속</button></div>
  </form>
</div></div>
"""

MAIN = """
<h1>6조 출입국</h1>
<p class="sub">실시간 출입 인원 현황을 확인하세요</p>
{% if msg %}<div class="msg {{ cls }}">{{ msg }}</div>{% endif %}
<div class="chips">
  <div class="status {{ 'ok' if running else '' }}">
    <i></i>
    {{ '모니터링 실행 중' if running else '모니터링 대기' }}
  </div>
</div>
{% if not script %}
<div class="msg err">{{ script_name }} 를 찾지 못했어요. 홈 폴더 아래에 파일이 있는지 확인해주세요.</div>
{% endif %}
<div class="tiles">
  <a class="tile set" href="{{ url_for('setcount') }}">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3"/><path d="M1 14h6M9 8h6M17 16h6"/></svg>
    <div><strong>초기 인원수 설정</strong><span>현재 실내 인원을 설정하세요</span></div>
  </a>
  <a class="tile go" href="{{ url_for('run') }}">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 4.5v15l13-7.5z" fill="currentColor"/></svg>
    <div><strong>실행</strong><span>실시간 출입 모니터링을 시작하세요</span></div>
  </a>
</div>
"""

SETCOUNT = """
<h1>초기 인원수 설정</h1>
<form class="panel" method="post">
  {% if msg %}<div class="msg {{ cls }}">{{ msg }}</div>{% endif %}
  <label for="total">인원수를 설정해주세요</label>
  <div class="stepper">
    <button type="button" id="minus" aria-label="1 감소">−</button>
    <input id="total" name="total" type="number" min="0" step="1" value="{{ val }}" required autofocus>
    <button type="button" id="plus" aria-label="1 증가">+</button>
  </div>
  <div class="actions">
    <button class="btn" type="submit">저장</button>
    <a class="btn ghost" href="{{ url_for('main') }}">돌아가기</a>
  </div>
</form>
<script>
(function(){
  var t=document.getElementById('total');
  function v(){var n=parseInt(t.value,10);return isNaN(n)?0:n;}
  document.getElementById('minus').onclick=function(){t.value=Math.max(0,v()-1);};
  document.getElementById('plus').onclick=function(){t.value=v()+1;};
})();
</script>
"""

RUN = """
<h1>실시간 출입 모니터링</h1>
{% if cls == 'err' %}
<div class="panel">
  <div class="msg err" style="margin-bottom:0">{{ msg }}</div>
  <div class="actions">
    <a class="btn" href="{{ url_for('run') }}">다시 실행</a>
    <a class="btn ghost" href="{{ url_for('main') }}">돌아가기</a>
    <a class="btn ghost" href="{{ url_for('log') }}" target="_blank">로그</a>
  </div>
</div>
{% else %}
<p class="sub">{{ msg or '카메라가 켜지는 데 몇 초 걸릴 수 있어요' }}</p>

<div class="stage">
  <div class="monitor-layout">
    <div class="viewport">
      <span class="br tl"></span>
      <span class="br tr"></span>
      <span class="br bl"></span>
      <span class="br brr"></span>

      <div class="tag" id="tag">
        <i></i>
        <span id="tagtxt">카메라 준비 중</span>
      </div>

      <div class="wait" id="wait">
        <div class="spin"></div>
        <span>영상을 불러오는 중…</span>
      </div>

      <img id="cam" src="{{ url_for('video') }}" alt="웹캠 화면">
    </div>

    <aside class="stats-panel">
      <section class="metric-card total-card">
        <div class="metric-label">현재 실내 인원</div>
        <div class="metric-value">
          <span id="n-total">–</span>
          <span class="metric-unit">명</span>
        </div>
      </section>

      <div class="io-grid">
        <section class="metric-card in-card">
          <div class="metric-label">입장 (IN)</div>
          <div class="metric-value">
            <span id="n-in">–</span>
            <span class="metric-unit">명</span>
          </div>
        </section>

        <section class="metric-card out-card">
          <div class="metric-label">퇴장 (OUT)</div>
          <div class="metric-value">
            <span id="n-out">–</span>
            <span class="metric-unit">명</span>
          </div>
        </section>
      </div>

      <section class="metric-card fps-card">
        <div class="metric-label">처리 속도</div>
        <div class="metric-value">
          <span id="n-fps">–</span>
          <span class="metric-unit">FPS</span>
        </div>
      </section>
    </aside>
  </div>

  <div class="run-actions">
    <form method="post" action="{{ url_for('stop') }}">
      <button class="btn danger" type="submit">■ 모니터링 중지</button>
    </form>
    <a class="btn ghost" href="{{ url_for('main') }}">돌아가기</a>
    <button class="btn ghost" type="button" id="logbtn">로그 보기</button>
  </div>

  <section class="inline-log-panel" id="inline-log-panel">
    <div class="inline-log-head">
      <strong>실행 로그</strong>
      <span>로그 내용이 2초마다 자동 갱신됩니다</span>
    </div>
    <pre id="inline-logbox">로그를 불러오는 중…</pre>
  </section>
</div>


<script>
(function(){
  var $=function(id){return document.getElementById(id);};
  var cam=$('cam'), everAlive=false;

  cam.addEventListener('load', function(){
    cam.classList.add('on');
    $('wait').style.display='none';
  });

  function bump(el, v){
    var s=(v===null||v===undefined)?'–':String(v);
    if(el.textContent!==s){
      el.textContent=s;
      el.classList.remove('pop');
      void el.offsetWidth;
      el.classList.add('pop');
    }
  }

  function setTag(cls, txt){
    $('tag').className='tag '+cls;
    $('tagtxt').textContent=txt;
  }

  async function poll(){
    try{
      var r=await fetch('{{ url_for("stats") }}', {cache:'no-store'});
      var d=await r.json();

      if(d.alive){
        everAlive=true;
        setTag('live','● LIVE');
      }else if(everAlive){
        setTag('off','정지됨');
      }else{
        setTag('','카메라 준비 중');
      }

      bump($('n-total'), d.total);
      bump($('n-in'), d['in']);
      bump($('n-out'), d.out);
      bump($('n-fps'), d.fps==null ? null : Number(d.fps).toFixed(1));
    }catch(e){}
  }

  poll();
  setInterval(poll, 1000);

  /* 로그: 팝업 대신 실행 화면 아래 패널을 열고 닫음 */
  var logBtn=$('logbtn');
  var inlinePanel=$('inline-log-panel');
  var inlineBox=$('inline-logbox');
  var logTimer=null;

  async function loadLog(){
    try{
      var r=await fetch('{{ url_for("log") }}', {cache:'no-store'});
      var t=await r.text();

      if(inlineBox){
        var atBottom=
          inlineBox.scrollTop+inlineBox.clientHeight>=inlineBox.scrollHeight-24;
        inlineBox.textContent=t;
        if(atBottom) inlineBox.scrollTop=inlineBox.scrollHeight;
      }
    }catch(e){}
  }

  function openInlineLog(){
    inlinePanel.classList.add('open');
    logBtn.textContent='로그 닫기';
    loadLog();
    if(logTimer===null){
      logTimer=setInterval(loadLog,2000);
    }

    // 버튼을 누르면 아래 로그 패널이 보이도록 이동
    setTimeout(function(){
      inlinePanel.scrollIntoView({behavior:'smooth', block:'nearest'});
    }, 0);
  }

  function closeInlineLog(){
    inlinePanel.classList.remove('open');
    logBtn.textContent='로그 보기';
    if(logTimer!==null){
      clearInterval(logTimer);
      logTimer=null;
    }
  }

  logBtn.onclick=function(){
    if(inlinePanel.classList.contains('open')){
      closeInlineLog();
    }else{
      openInlineLog();
    }
  };

})();
</script>
{% endif %}
"""


def page(body_tpl, hb=True, wide=False, **ctx):
    body = render_template_string(body_tpl, host=PI_HOST, **ctx)
    return render_template_string(BASE, body=body, host=PI_HOST,
                                  user=current_user(), hb=hb, wide=wide)


# ---------------------------------------------------------------------------
# SSH 헬퍼
# ---------------------------------------------------------------------------
def current():
    tok = session.get("tok")
    return SESSIONS.get(tok) if tok else None


def current_user():
    c = current()
    return c["user"] if c else None


def get_client():
    c = current()
    if not c:
        return None
    tr = c["client"].get_transport()
    if not tr or not tr.is_active():
        SESSIONS.pop(session.get("tok"), None)
        return None
    return c["client"]


def ssh_exec(client, cmd, timeout=20):
    """홈디렉터리에서 bash로 실행"""
    full = "cd ~ && " + cmd
    _, out, err = client.exec_command("bash -lc " + shlex.quote(full), timeout=timeout)
    o, e = out.read().decode(errors="replace"), err.read().decode(errors="replace")
    return out.channel.recv_exit_status(), o, e


# 'stream_runner.py' 를 '[s]tream_runner.py' 로 쓰는 이유: pkill -f 가 자기 자신(bash -lc ...)의
# 명령줄과 매칭되어 셸이 같이 죽는 문제를 피하기 위함.
STOP_CMD = r"""
PIDF=__PIDF__
if [ -f "$PIDF" ] && grep -aq 'stream_[r]unner' "/proc/$(cat "$PIDF")/cmdline" 2>/dev/null; then
  PID=$(cat "$PIDF")
  kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
  for i in 1 2 3 4 5 6; do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
  kill -KILL -- "-$PID" 2>/dev/null
  kill -KILL "$PID" 2>/dev/null
fi
rm -f "$PIDF"
pkill -TERM -u "$(id -un)" -f '[s]tream_runner.py' 2>/dev/null
sleep 0.5
pkill -KILL -u "$(id -un)" -f '[s]tream_runner.py' 2>/dev/null
rm -f __FRAME__ __FRAME__.tmp __STATS__ __STATS__.tmp
if pgrep -u "$(id -un)" -f '[s]tream_runner.py' >/dev/null; then echo RUNNING; else echo STOPPED; fi
""".replace("__PIDF__", PID_PATH).replace("__FRAME__", FRAME_PATH).replace("__STATS__", STATS_PATH)

STATUS_CMD = ("if pgrep -u \"$(id -un)\" -f '[s]tream_runner.py' >/dev/null; "
              "then echo RUNNING; else echo STOPPED; fi")


def stop_runner(client):
    """door_counter(래퍼 포함) 프로세스 그룹을 종료. 종료되었으면 True"""
    try:
        if client is not None:
            tr = client.get_transport()
            if tr and tr.is_active():
                _, out, _ = ssh_exec(client, STOP_CMD, timeout=30)
                return "STOPPED" in out
    except Exception:
        pass
    # SSH 연결이 끊긴 경우: 같은 계정으로 실행 중이라면 로컬에서라도 종료 시도
    try:
        subprocess.run(["pkill", "-TERM", "-f", "[s]tream_runner.py"], timeout=5)
        time.sleep(0.5)
        subprocess.run(["pkill", "-KILL", "-f", "[s]tream_runner.py"], timeout=5)
        for p in (FRAME_PATH, STATS_PATH, PID_PATH):
            try:
                os.remove(p)
            except OSError:
                pass
    except Exception:
        pass
    return False


def runner_status(client):
    try:
        _, out, _ = ssh_exec(client, STATUS_CMD, timeout=10)
        return "● 실행 중" if "RUNNING" in out else "■ 정지됨"
    except Exception:
        return "(상태 확인 실패)"


def login_required(fn):
    from functools import wraps

    @wraps(fn)
    def w(*a, **k):
        if not get_client():
            return redirect(url_for("login"))
        return fn(*a, **k)
    return w


def find_script(client):
    """홈디렉터리 아래에서 door_counter.py 를 찾아 절대경로를 반환 (없으면 None).
    여러 개면 CNN_Team_Project 경로 우선, 그다음 가장 최근 수정된 것."""
    cmd = (f'find "$HOME" -type f -name {SCRIPT_NAME} '
           '-not -path "*/.cache/*" -not -path "*/site-packages/*" '
           '-not -path "*/.local/*" -printf "%T@ %p\\n" 2>/dev/null | sort -rn')
    _, out, _ = ssh_exec(client, cmd, timeout=60)
    paths = [ln.split(" ", 1)[1] for ln in out.splitlines() if " " in ln]
    if not paths:
        return None
    preferred = [p for p in paths if "CNN_Team_Project" in p]
    return (preferred or paths)[0]


def get_script(client):
    c = current()
    if not c.get("script"):
        c["script"] = find_script(client)
    return c["script"]


def find_total_line(lines):
    """'total = 숫자' 줄을 자동 검색. 여러 개면 들여쓰기가 가장 얕은(전역) 줄, 그다음 첫 줄.
    반환: (0-based 인덱스, 후보 개수) 또는 (None, 0)"""
    cands = []
    for i, ln in enumerate(lines):
        m = TOTAL_RE.match(ln)
        if m:
            indent = len(ln) - len(ln.lstrip())
            cands.append((indent, i))
    if not cands:
        return None, 0
    cands.sort()
    return cands[0][1], len(cands)


def masker_control(cmd, value=None, timeout=1.5):
    """실행 중인 masker.py의 6200 제어 포트에 JSON 명령을 보냄."""
    req = {"cmd": cmd}
    if value is not None:
        req["value"] = int(value)

    with socket.create_connection((CONTROL_HOST, CONTROL_PORT), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))

        buf = bytearray()
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf.extend(chunk)

    if not buf:
        raise ConnectionError("masker.py 제어 포트 응답 없음")

    return json.loads(bytes(buf).split(b"\n", 1)[0].decode("utf-8"))


def total_file_path(script_path):
    """masker.py와 같은 폴더의 door_total.json 경로."""
    return os.path.join(os.path.dirname(script_path), "door_total.json")


def read_initial_total(client, script_path):
    """현재 저장된 인원수 읽기.
    우선순위: 실행 중 masker -> door_total.json -> TOTAL_START
    """
    # 1) 실행 중 masker.py 값
    try:
        resp = masker_control("get")
        if resp.get("ok"):
            return max(0, int(resp["total"]))
    except Exception:
        pass

    sftp = client.open_sftp()
    try:
        # 2) door_total.json 값
        try:
            with sftp.open(total_file_path(script_path), "r") as f:
                data = json.loads(f.read().decode("utf-8"))
            return max(0, int(data["total"]))
        except Exception:
            pass

        # 3) fallback: TOTAL_START
        try:
            with sftp.open(script_path, "r") as f:
                source = f.read().decode("utf-8")
            lines = source.split("\n")
            idx, _ = find_total_line(lines)
            if idx is not None:
                m = TOTAL_RE.match(lines[idx])
                if m:
                    return max(0, int(m.group(2)))
        except Exception:
            pass
    finally:
        sftp.close()

    return 0


def save_initial_total(client, script_path, value):
    """웹에서 설정한 인원수를 실제 masker가 쓰는 값에 반영.

    1) door_total.json 저장 -> 다음 실행 때 이 값으로 시작
    2) TOTAL_START도 같은 값으로 동기화 -> 저장 파일이 없을 때 fallback
    3) masker.py가 이미 실행 중이면 set_total로 즉시 반영
       (masker 쪽에서 IN/OUT도 0으로 초기화)
    """
    value = max(0, int(value))
    sftp = client.open_sftp()

    try:
        # 1) door_total.json 저장
        total_path = total_file_path(script_path)
        tmp_path = total_path + ".tmp"
        payload = json.dumps(
            {"total": value, "ts": time.time()},
            ensure_ascii=False
        ).encode("utf-8")

        with sftp.open(tmp_path, "w") as f:
            f.write(payload)

        try:
            sftp.remove(total_path)
        except OSError:
            pass
        sftp.rename(tmp_path, total_path)

        # 2) masker.py의 TOTAL_START도 같이 동기화
        with sftp.open(script_path, "r") as f:
            source = f.read().decode("utf-8")

        lines = source.split("\n")
        idx, _ = find_total_line(lines)
        if idx is not None:
            old_line = lines[idx]
            new_line = TOTAL_RE.sub(
                lambda m: m.group(1) + str(value),
                old_line,
                count=1
            )
            if new_line != old_line:
                # 백업은 최초 1회만
                try:
                    sftp.stat(script_path + ".bak")
                except OSError:
                    with sftp.open(script_path + ".bak", "w") as f:
                        f.write(source.encode("utf-8"))

                lines[idx] = new_line
                with sftp.open(script_path, "w") as f:
                    f.write("\n".join(lines).encode("utf-8"))
    finally:
        sftp.close()

    # 3) 이미 실행 중이면 메모리 값도 즉시 변경
    live_applied = False
    try:
        resp = masker_control("set_total", value)
        live_applied = bool(resp.get("ok"))
    except Exception:
        # 실행 전이면 포트가 닫혀 있는 것이 정상
        pass

    return live_applied


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return page(LOGIN, error=None)

    user = request.form.get("username", "").strip()
    pw = request.form.get("password", "")
    cl = paramiko.SSHClient()
    cl.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        cl.connect(PI_HOST, port=PI_PORT, username=user, password=pw,
                   timeout=10, allow_agent=False, look_for_keys=False)
    except paramiko.AuthenticationException:
        return page(LOGIN, error="username 또는 비밀번호가 올바르지 않습니다.")
    except Exception as e:
        return page(LOGIN, error=f"SSH 연결 실패: {e}")

    # 접속 즉시 홈에서 가상환경 activate (성공 여부 확인)
    code, out, err = ssh_exec(cl, f"source {VENV_ACTIVATE} && python -V")
    venv = (f"✅ yolo-env 활성화됨 ({(out or err).strip()})" if code == 0
            else f"⚠️ yolo-env 활성화 실패: {err.strip()}")

    script = find_script(cl)   # door_counter.py 자동 검색
    tok = secrets.token_hex(16)
    SESSIONS[tok] = {"client": cl, "user": user, "venv": venv, "script": script}
    session["tok"] = tok
    return redirect(url_for("main"))


@app.route("/main")
@login_required
def main():
    cl = get_client()
    script = get_script(cl)
    flash = session.pop("flash", None) or (None, None)
    raw = current()["venv"]
    venv_ok = raw.startswith("✅")
    venv_txt = raw.lstrip("✅⚠️ ").strip()
    running = "실행 중" in runner_status(cl)
    return page(MAIN, venv=venv_txt, venv_ok=venv_ok, script=script,
                script_name=SCRIPT_NAME, running=running,
                msg=flash[0], cls=flash[1])


@app.route("/setcount", methods=["GET", "POST"])
@login_required
def setcount():
    cl = get_client()
    path = get_script(cl)

    if not path:
        return page(
            SETCOUNT,
            path="",
            line="-",
            current="",
            val="0",
            msg=f"홈디렉터리에서 {SCRIPT_NAME} 를 찾지 못했습니다.",
            cls="err"
        )

    msg = cls = None

    if request.method == "POST":
        raw = request.form.get("total", "").strip()

        if not re.fullmatch(r"\d+", raw):
            val = read_initial_total(cl, path)
            return page(
                SETCOUNT,
                path=path,
                line="-",
                current="",
                val=str(val),
                msg="0 이상의 정수를 입력해주세요.",
                cls="err"
            )

        value = int(raw)

        try:
            live_applied = save_initial_total(cl, path, value)

            if live_applied:
                msg = f"✓ 현재 인원이 {value}명으로 즉시 설정되었습니다. (IN/OUT 0으로 초기화)"
            else:
                msg = f"✓ 초기 인원수가 {value}명으로 저장되었습니다. 다음 실행부터 적용됩니다."
            cls = "ok"
        except Exception as e:
            msg = f"저장 실패: {e}"
            cls = "err"

    val = read_initial_total(cl, path)

    return page(
        SETCOUNT,
        path=path,
        line="-",
        current=f"door_total.json: {val}",
        val=str(val),
        msg=msg,
        cls=cls
    )


@app.route("/run")
@login_required
def run():
    cl = get_client()
    script = get_script(cl)
    if not script:
        return page(RUN, wide=True, msg=f"홈디렉터리에서 {SCRIPT_NAME} 를 찾지 못했습니다.", cls="err")
    try:
        stop_runner(cl)                     # 이전 실행이 남아 있으면 먼저 종료 (프레임/통계 파일도 삭제)

        sftp = cl.open_sftp()
        with sftp.open(RUNNER_PATH, "w") as f:
            f.write(RUNNER_CODE.encode())
        sftp.close()
        # 홈에서 venv activate 후 python ./.../door_counter.py 실행(래퍼 경유).
        # setsid: 별도 프로세스 그룹으로 띄워 나중에 그룹 전체를 한 번에 종료할 수 있게 함.
        busy_wait_env = "1" if BUSY_WAIT else "0"
        cmd = (f"source {VENV_ACTIVATE} && "
               f"({THREAD_ENV} DOOR_BUSY_WAIT={busy_wait_env} setsid nohup python {RUNNER_PATH} {shlex.quote(script)} "
               f"> {LOG_PATH} 2>&1 & echo $! > {PID_PATH}); "
               f"sleep 1.5; "
               f"if kill -0 $(cat {PID_PATH}) 2>/dev/null; then echo OK; else echo DEAD; fi")
        _, out, err = ssh_exec(cl, cmd, timeout=30)
        if "OK" in out:
            msg, cls = "모니터링이 실행 중입니다", "ok"
        else:
            _, tail, _ = ssh_exec(cl, f"tail -n 15 {LOG_PATH} 2>/dev/null")
            msg, cls = f"실행 직후 종료되었습니다.\n{tail or err}", "err"
    except Exception as e:
        msg, cls = f"실행 실패: {e}", "err"
    return page(RUN, wide=True, msg=msg, cls=cls)


@app.route("/stop", methods=["POST"])
@login_required
def stop():
    ok = stop_runner(get_client())
    session["flash"] = (("✓ 모니터링이 중지되었습니다.", "ok") if ok
                         else ("모니터링 중지를 확인하지 못했습니다. 로그를 확인해주세요.", "err"))
    return redirect(url_for("main"))


@app.route("/stats")
@login_required
def stats():
    """실행 화면이 1초마다 호출: 동작 여부 + 현재 인원 / IN / OUT / FPS"""
    try:
        alive = time.time() - os.path.getmtime(FRAME_PATH) < 3
    except FileNotFoundError:
        alive = False

    data = {}
    try:
        with open(STATS_PATH) as f:
            data = json.load(f)
    except Exception:
        pass

    # masker.py 실행 로그의 마지막 부분에서 FPS와
    # 학생 수 / IN / OUT의 최신 상태를 읽는다.
    fps = None
    tail = ""

    try:
        with open(LOG_PATH, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode(errors="replace")

        matches = re.findall(r"FPS=\s*([0-9]+(?:\.[0-9]+)?)", tail)
        if matches:
            fps = float(matches[-1])
    except Exception:
        pass

    return jsonify({
        "alive": alive,
        "total": data.get("total"),
        "in": data.get("in"),
        "out": data.get("out"),
        "fps": fps,
    })

@app.route("/log")
@login_required
def log():
    _, out, _ = ssh_exec(get_client(), f"tail -n 60 {LOG_PATH} 2>/dev/null")
    return Response(out or "(로그 없음)", mimetype="text/plain; charset=utf-8")


@app.route("/video")
@login_required
def video():
    def gen():
        last, data, sent_at = None, None, 0
        while not STOPPING.is_set():
            try:
                m = os.path.getmtime(FRAME_PATH)
                if m != last:
                    last = m
                    with open(FRAME_PATH, "rb") as f:
                        data = f.read()
                    sent_at = 0
            except FileNotFoundError:
                pass
            now = time.time()
            # 새 프레임이 있거나 2초 이상 정체되면 전송 -> 브라우저가 닫혔을 때 쓰기 오류로 스레드 종료
            if data and (sent_at == 0 or now - sent_at > 2):
                sent_at = now
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------------------
# 종료 / 포트 반환
# ---------------------------------------------------------------------------
STOPPING = threading.Event()
HB = {"last": None, "leaving": None}


def cleanup_all():
    """모든 세션의 door_counter 종료 + SSH 연결 닫기 (여러 번 호출해도 안전)"""
    items = list(SESSIONS.items())
    SESSIONS.clear()
    if not items:
        stop_runner(None)       # 세션이 없어도 남은 프로세스가 있으면 정리 시도
    for _, c in items:
        try:
            stop_runner(c["client"])
        except Exception:
            pass
        try:
            c["client"].close()
        except Exception:
            pass


def shutdown_server(reason=""):
    if STOPPING.is_set():
        return
    STOPPING.set()
    print(f"[종료] {reason} -> door_counter 종료, 웹 서버 종료(포트 {WEB_PORT} 반환)", flush=True)
    cleanup_all()
    t = threading.Timer(5, lambda: os._exit(0))   # 정상 종료가 안 될 때의 안전장치
    t.daemon = True
    t.start()
    try:
        os.kill(os.getpid(), signal.SIGINT)        # app.run() 정상 종료 -> 소켓 close
    except Exception:
        os._exit(0)


def watchdog():
    while not STOPPING.is_set():
        time.sleep(2)
        last, leaving = HB["last"], HB["leaving"]
        if last is None:
            continue                                # 아직 아무도 접속 안 함
        now = time.time()
        if leaving and now - leaving > LEAVE_GRACE:
            shutdown_server("웹페이지 닫힘 감지")
        elif now - last > HB_TIMEOUT:
            shutdown_server(f"{HB_TIMEOUT}초 동안 브라우저 응답 없음")


@app.route("/ping", methods=["POST"])
def ping():
    HB["last"] = time.time()
    HB["leaving"] = None
    return ("", 204)


@app.route("/leaving", methods=["POST"])
def leaving():
    HB["leaving"] = time.time()    # 새로고침/페이지 이동이면 곧바로 /ping 이 와서 취소됨
    return ("", 204)


@app.route("/logout", methods=["POST"])
def logout():
    """로그아웃: door_counter.py 종료 + SSH 연결 닫기. 서버/포트는 유지."""
    c = SESSIONS.pop(session.pop("tok", None), None)
    if c:
        stop_runner(c["client"])
        try:
            c["client"].close()
        except Exception:
            pass
    return redirect(url_for("login"))


def _on_signal(signum, frame):
    cleanup_all()
    sys.exit(0)


def port_in_use(port):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return False
        except OSError:
            return True


if __name__ == "__main__":
    if port_in_use(WEB_PORT):
        print(f"[오류] 포트 {WEB_PORT} 를 이미 다른 프로세스가 사용 중입니다.\n"
              f"  이전에 실행한 app.py 가 남아 있다면:  fuser -k {WEB_PORT}/tcp\n"
              f"  또는 다른 포트로 실행:  WEB_PORT=8001 python3 app.py")
        sys.exit(1)
    for sig in (signal.SIGTERM, signal.SIGHUP):    # kill / 터미널 닫힘
        signal.signal(sig, _on_signal)
    signal.signal(signal.SIGINT, signal.default_int_handler)  # nohup/& 로 실행해도 종료 신호가 먹히도록
    atexit.register(cleanup_all)
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        app.run(host="0.0.0.0", port=WEB_PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        STOPPING.set()
        cleanup_all()
        print(f"서버 종료 완료 - 포트 {WEB_PORT} 반환됨")
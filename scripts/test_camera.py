#!/usr/bin/env python
"""Passive Go2 front-camera test: reads the H.264 RTP multicast stream and saves frames.
Sends nothing to the robot. Needs OpenCV with FFmpeg.
Usage: python scripts/test_camera.py [--iface-ip 192.168.123.222] [--seconds 6] [--out output]"""
import argparse, os, socket, time
import cv2, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--iface-ip", default="192.168.123.222")
p.add_argument("--group", default="230.1.1.1")
p.add_argument("--port", type=int, default=1720)
p.add_argument("--seconds", type=float, default=6)
p.add_argument("--out", default="output")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
s.bind(("", a.port))
s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
             socket.inet_aton(a.group) + socket.inet_aton(a.iface_ip))
s.settimeout(3)

SC = b"\0\0\0\1"
stream, seen_sps, pkts, stamps = bytearray(), False, 0, set()
t0 = time.time()
while time.time() - t0 < a.seconds:
    d, _ = s.recvfrom(65535)
    if d[0] >> 6 != 2:
        continue
    off = 12 + 4 * (d[0] & 15)                      # skip RTP header + CSRCs
    if d[0] & 0x10:                                 # skip header extension
        off += 4 + 4 * int.from_bytes(d[off + 2:off + 4], "big")
    pl = d[off:]
    pkts += 1; stamps.add(d[4:8])
    nal = pl[0] & 31
    if nal == 7:
        seen_sps = True
    if not seen_sps and nal != 28:
        continue
    if nal == 28:                                   # FU-A fragment
        if pl[1] & 0x80:
            if (pl[1] & 31) == 5 or seen_sps:
                stream += SC + bytes([(pl[0] & 0xE0) | (pl[1] & 31)]) + pl[2:]
        elif seen_sps:
            stream += pl[2:]
    elif nal == 24:                                 # STAP-A aggregate
        i = 1
        while i + 2 <= len(pl):
            n = int.from_bytes(pl[i:i + 2], "big"); i += 2
            stream += SC + pl[i:i + n]; i += n
    elif seen_sps:
        stream += SC + pl
dt = time.time() - t0
path = os.path.join(a.out, "camera.h264")
open(path, "wb").write(stream)
print(f"RX {pkts} pkts, {len(stamps)} frames, {len(stamps)/dt:.1f} fps, {len(stream)*8/dt/1e6:.2f} Mbps")

cap = cv2.VideoCapture(path)
n, last = 0, None
while True:
    ok, f = cap.read()
    if not ok:
        break
    n += 1; last = f
if last is None:
    raise SystemExit("FAIL: no frames decoded")
img = os.path.join(a.out, "camera_frame.jpg")
cv2.imwrite(img, last)
g = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY)
print(f"Decoded {n} frames, {last.shape[1]}x{last.shape[0]}, brightness={g.mean():.0f}, "
      f"contrast={g.std():.0f}, sharpness={cv2.Laplacian(g, cv2.CV_64F).var():.0f}")
print(f"PASS: saved {img}")

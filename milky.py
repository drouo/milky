"""
Milky  -  a MILKDROP-style music visualizer for the terminal generation.

Pure pygame + numpy, with optional live-audio input.  It recreates the
classic Milkdrop feel -- a warping feedback tunnel, reactive waveforms,
bloom glow, kaleidoscopic mirrors and auto-morphing presets -- and adds a
big pile of scenes and live options on top.

    python milky.py                 # synthetic beat engine, windowed
    python milky.py --audio         # react to your microphone (needs sounddevice)
    python milky.py --list          # list presets and exit
    python milky.py -W 1280 -H 720 --fullscreen

Keys (also printed in the HUD):
    Esc / q      quit                 space    next preset
    r            random preset        a        toggle auto-cycle
    n / p        next / prev scene    c        cycle color palette
    m            cycle mirror mode    k        toggle kaleidoscope
    b            toggle bloom         t        toggle beat flash
    [ / ]        decay down / up      - / =    zoom down / up
    up / down    speed                s        save screenshot (PNG)
    h            hide / show HUD      f        toggle fullscreen

Requires:  pip install pygame numpy    (optional: sounddevice)
"""

import os
import sys
import math
import time
import random
import argparse
from collections import deque

try:
    import pygame
    import numpy as np
except ImportError as e:  # pragma: no cover
    print("Missing dependency:", e)
    print("Install with:  pip install pygame numpy")
    sys.exit(1)

try:
    import sounddevice as sd
except Exception:
    sd = None

TAU = math.tau


# --------------------------------------------------------------------------
# color / palettes
# --------------------------------------------------------------------------
def hsv2rgb(h, s, v):
    """Fast HSV->RGB (all in [0,1]) -> (r,g,b) 0-255."""
    h = h % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r, g, b = (
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    )[i % 6]
    return (int(r * 255), int(g * 255), int(b * 255))


# palettes map a value 0..1 (+ a drifting base hue) to a color.
# h0/h1 = hue window, sat = saturation.  "mono" collapses the window.
PALETTES = [
    dict(name="Rainbow", h0=0.0, h1=1.0, sat=0.85),
    dict(name="Fire",    h0=0.98, h1=1.18, sat=0.95),
    dict(name="Ice",     h0=0.45, h1=0.68, sat=0.80),
    dict(name="Toxic",   h0=0.20, h1=0.42, sat=0.95),
    dict(name="Sunset",  h0=0.86, h1=1.12, sat=0.88),
    dict(name="Neon",    h0=0.70, h1=0.98, sat=1.00),
    dict(name="Mono",    h0=0.55, h1=0.60, sat=0.15),
]


class Palette:
    def __init__(self, i=0):
        self.i = i

    def cur(self):
        return PALETTES[self.i % len(PALETTES)]

    def next(self):
        self.i = (self.i + 1) % len(PALETTES)

    def color(self, frac, base, v=1.0):
        p = self.cur()
        h = p["h0"] + (p["h1"] - p["h0"]) * (frac % 1.0) + base
        return hsv2rgb(h, p["sat"], v)


# --------------------------------------------------------------------------
# audio: synthetic beat engine OR live microphone (sounddevice)
# --------------------------------------------------------------------------
class AudioEngine:
    N = 512          # waveform samples
    BINS = 64        # spectrum bars

    def __init__(self, live=False):
        self.live = bool(live and sd is not None)
        self.x = np.linspace(0.0, 1.0, self.N, dtype=np.float32)
        self.bass = self.mid = self.treble = 0.0
        self.beat = 0.0
        self.hit = False
        self.bpm = 125.0
        self.spec = np.zeros(self.BINS, dtype=np.float32)
        self._avg = 0.0
        self._buf = np.zeros(self.N, dtype=np.float32)
        self._stream = None
        if self.live:
            try:
                self._stream = sd.InputStream(
                    channels=1, samplerate=44100, blocksize=self.N,
                    callback=self._cb)
                self._stream.start()
            except Exception as e:
                print("live audio failed, using synthetic:", e)
                self.live = False

    def _cb(self, indata, frames, timeinfo, status):
        m = indata[:, 0]
        if len(m) >= self.N:
            self._buf = m[-self.N:].astype(np.float32)
        else:
            self._buf = np.roll(self._buf, -len(m))
            self._buf[-len(m):] = m

    def _spectrum(self, wave):
        mag = np.abs(np.fft.rfft(wave * np.hanning(len(wave))))
        mag = mag[: self.BINS * 2]
        if len(mag) < self.BINS:
            mag = np.pad(mag, (0, self.BINS - len(mag)))
        mag = mag[: self.BINS]
        mag = np.log1p(mag) / 4.0
        # attack fast, release slow -> lively bars
        self.spec = np.maximum(mag, self.spec * 0.86)
        return np.clip(self.spec, 0, 1)

    def sample(self, t, dt):
        if self.live:
            wave = self._buf.copy()
            mx = float(np.max(np.abs(wave))) or 1.0
            wave = wave / mx
            spec = self._spectrum(wave)
            self.bass = float(np.mean(spec[:8]))
            self.mid = float(np.mean(spec[8:28]))
            self.treble = float(np.mean(spec[28:]))
            self._avg += (self.bass - self._avg) * 0.05
            self.hit = self.bass > self._avg * 1.6 + 0.02
        else:
            period = 60.0 / self.bpm
            phase = (t % period) / period
            env = math.exp(-phase * 5.0)
            self.hit = phase < (dt / period) * 1.5
            fill = 0.0
            if (t % 8.0) > 6.0:
                fill = math.exp(-((t * 4) % 1.0) * 6.0) * 0.7
            self.bass = max(env, fill)
            self.mid = 0.5 + 0.5 * math.sin(t * 3.1)
            self.treble = 0.5 + 0.5 * math.sin(t * 11.3 + math.sin(t))
            x = self.x
            wave = (
                np.sin(x * TAU * 3 + t * 2.0) * self.bass
                + np.sin(x * TAU * 7 + t * 3.0) * self.mid * 0.5
                + np.sin(x * TAU * 15 + t * 5.0) * self.treble * 0.3
                + np.sin(x * TAU * 31 + t * 9.0) * self.treble * 0.12
            ).astype(np.float32)
            wave *= (0.55 + 0.9 * self.bass)
            spec = self._spectrum(wave)

        self.beat += (self.bass - self.beat) * min(1.0, dt * 12.0)
        return wave, spec

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
# starfield state (used by the starfield scene)
# --------------------------------------------------------------------------
class Stars:
    def __init__(self, n, w, h):
        self.w, self.h = w, h
        self.p = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
        self.p[:, 2] = np.random.uniform(0.05, 1.0, n)

    def step(self, speed):
        self.p[:, 2] -= speed
        gone = self.p[:, 2] < 0.02
        self.p[gone, 2] = 1.0
        self.p[gone, 0] = np.random.uniform(-1, 1, gone.sum())
        self.p[gone, 1] = np.random.uniform(-1, 1, gone.sum())


# --------------------------------------------------------------------------
# particle system (beat bursts)
# --------------------------------------------------------------------------
class Particles:
    MAX = 900

    def __init__(self):
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.life = np.zeros((0,), np.float32)
        self.hue = np.zeros((0,), np.float32)

    def burst(self, x, y, n, spread, hue):
        n = min(n, self.MAX)
        ang = np.random.uniform(0, TAU, n)
        spd = np.random.uniform(0.2, 1.0, n) * spread
        v = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1).astype(np.float32)
        p = np.full((n, 2), (x, y), np.float32)
        self.pos = np.concatenate([self.pos, p])[-self.MAX:]
        self.vel = np.concatenate([self.vel, v])[-self.MAX:]
        self.life = np.concatenate([self.life, np.ones(n, np.float32)])[-self.MAX:]
        self.hue = np.concatenate([self.hue, np.full(n, hue, np.float32)])[-self.MAX:]

    def step(self, dt):
        if len(self.pos):
            self.pos += self.vel * dt * 60
            self.vel *= 0.97
            self.life -= dt * 0.9
            keep = self.life > 0
            self.pos, self.vel = self.pos[keep], self.vel[keep]
            self.life, self.hue = self.life[keep], self.hue[keep]


class DinoGame:
    def __init__(self, rw, rh):
        self.rw, self.rh = rw, rh
        self.gy = rh - 50
        self.dx, self.dy = 100, self.gy
        self.dv = 0.0
        self.jumping = False
        self.frame = 0.0
        self.obstacles: list[dict] = []
        self.score = 0.0
        self.spd = 3.0
        self._timer = 0.0
        self._bounce_timer = 0.0

    def reset(self):
        self.dy = self.gy
        self.dv = 0.0
        self.jumping = False
        self.obstacles.clear()
        self.score = 0.0
        self.spd = 3.0
        self._timer = 0.0
        self._bounce_timer = 0.0

    def step(self, beat_hit, dt):
        s = dt * 60
        if beat_hit and not self.jumping:
            self.dv = -14
            self.jumping = True
        self._bounce_timer -= dt
        if self._bounce_timer <= 0 and not self.jumping:
            self.dv = -14
            self.jumping = True
            self._bounce_timer = random.uniform(0.8, 2.0)
        self.dv += 0.4 * s
        self.dy += self.dv * s
        if self.dy >= self.gy:
            self.dy = self.gy
            self.dv = 0.0
            self.jumping = False

        self._timer -= dt
        if self._timer <= 0:
            self._timer = max(0.3, random.uniform(0.6, 2.0) - self.score * 0.003)
            if random.random() < 0.7:
                self.obstacles.append({"x": self.rw + 20, "h": random.choice((20, 30, 40)), "t": "cactus"})
            else:
                self.obstacles.append({"x": self.rw + 80, "h": 15, "t": "bird", "y": self.gy - 40 - random.random() * 30})

        for o in self.obstacles:
            o["x"] -= self.spd * dt * 60
        self.obstacles = [o for o in self.obstacles if o["x"] > -60]

        for o in self.obstacles:
            if abs(self.dx - o["x"]) < 22:
                if o["t"] == "bird":
                    top = o["y"] - 8
                    bot = o["y"] + 8
                    if self.dy > top and self.dy < bot:
                        self.obstacles.clear()
                        self.score = max(0, self.score - 5)
                else:
                    if self.gy - self.dy < o["h"]:
                        self.obstacles.clear()
                        self.score = max(0, self.score - 3)

        self.score += dt * self.spd * 2
        self.spd = 3.0 + self.score * 0.02
        self.frame += dt * 8

    def draw(self, canvas, col, scale=1.0):
        gy = self.gy
        sw = int(self.rw * scale)
        sh = int(self.rh * scale)
        pygame.draw.line(canvas, (60, 60, 60), (0, int(gy * scale)), (sw, int(gy * scale)), 2)

        leg_swing = int(math.sin(self.frame) * 4) if not self.jumping else 2
        body_c = col(0.15, 0.8)
        dark = (max(0, body_c[0] - 40), max(0, body_c[1] - 40), max(0, body_c[2] - 40))
        white = (240, 240, 240)

        # body
        body_r = pygame.Rect(int((self.dx - 18) * scale), int((self.dy - 24) * scale),
                             int(32 * scale), int(18 * scale))
        pygame.draw.ellipse(canvas, body_c, body_r)
        # head
        pygame.draw.circle(canvas, body_c, (int((self.dx + 12) * scale), int((self.dy - 30) * scale)),
                           int(9 * scale))
        # eye
        pygame.draw.circle(canvas, white, (int((self.dx + 15) * scale), int((self.dy - 32) * scale)),
                           int(3 * scale))
        pygame.draw.circle(canvas, (0, 0, 0), (int((self.dx + 16) * scale), int((self.dy - 32) * scale)),
                           int(1.5 * scale))
        # mouth
        pygame.draw.line(canvas, dark, (int((self.dx + 18) * scale), int((self.dy - 26) * scale)),
                         (int((self.dx + 22) * scale), int((self.dy - 26) * scale)), 2)
        # legs
        lw = max(1, int(4 * scale))
        pygame.draw.line(canvas, dark,
                         (int((self.dx - 10) * scale), int(self.dy * scale)),
                         (int((self.dx - 12 - leg_swing) * scale), int((self.dy + 14) * scale)), lw)
        pygame.draw.line(canvas, dark,
                         (int((self.dx + 4) * scale), int(self.dy * scale)),
                         (int((self.dx + 6 + leg_swing) * scale), int((self.dy + 14) * scale)), lw)
        # tail
        tail_pts = [(int((self.dx - 18) * scale), int((self.dy - 12) * scale)),
                    (int((self.dx - 28) * scale), int((self.dy - 8) * scale)),
                    (int((self.dx - 30) * scale), int((self.dy - 14) * scale))]
        pygame.draw.lines(canvas, body_c, False, tail_pts, max(1, int(3 * scale)))
        # spikes on back
        for i in range(4):
            sx = int((self.dx - 5 + i * 7) * scale)
            sy = int((self.dy - 24) * scale)
            pygame.draw.polygon(canvas, dark,
                                [(sx, sy), (int(sx + 4 * scale), int(sy - 7 * scale)),
                                 (int(sx + 7 * scale), int(sy))])

        # obstacles
        for o in self.obstacles:
            x, h = int(o["x"] * scale), int(o["h"] * scale)
            if o["t"] == "cactus":
                cact_c = (50, 160, 50)
                pygame.draw.rect(canvas, cact_c, (x - int(6 * scale), int(gy * scale) - h,
                                                  int(12 * scale), h))
                pygame.draw.rect(canvas, cact_c, (x - int(10 * scale), int(gy * scale) - h - int(10 * scale),
                                                  int(20 * scale), int(10 * scale)))
                arm_y = int(gy * scale) - h + int(10 * scale)
                pygame.draw.rect(canvas, cact_c, (x - int(14 * scale), arm_y - int(4 * scale),
                                                  int(8 * scale), int(8 * scale)))
                pygame.draw.rect(canvas, cact_c, (x + int(6 * scale), arm_y - int(4 * scale),
                                                  int(8 * scale), int(8 * scale)))
            else:
                bird_c = (180, 100, 60)
                by = int(o["y"] * scale)
                wing_up = math.sin(self.frame * 2 + o["x"] * 0.1) > 0
                ws = int(16 * scale)
                hs = int(10 * scale)
                if wing_up:
                    pygame.draw.polygon(canvas, bird_c, [(x, by), (x - ws, by - hs), (x - int(6 * scale), by)])
                    pygame.draw.polygon(canvas, bird_c, [(x, by), (x + ws, by - hs), (x + int(6 * scale), by)])
                else:
                    pygame.draw.polygon(canvas, bird_c, [(x, by), (x - int(12 * scale), by + int(4 * scale)),
                                                         (x - int(4 * scale), by)])
                    pygame.draw.polygon(canvas, bird_c, [(x, by), (x + int(12 * scale), by + int(4 * scale)),
                                                         (x + int(4 * scale), by)])

        # score
        font = pygame.font.SysFont("consolas", max(10, int(16 * scale)))
        txt = font.render(f"DINO {int(self.score)}m", True, (180, 200, 180))
        canvas.blit(txt, (int(10 * scale), int((self.rh - 80) * scale)))


# --------------------------------------------------------------------------
# dance party: geometric animals
# --------------------------------------------------------------------------

class PartyScene:
    N = 14       # animals
    GROUPS = 4   # dance groups

    # animal definitions: list of (type_name, hue_base, shapes)
    # shape commands: ("circle", cx, cy, r), ("ellipse", cx, cy, rx, ry),
    #                 ("line", x1, y1, x2, y2, w), ("poly", [(x1,y1),...])
    ANIMALS = [
        ("elephant", 0.00, [
            ("ellipse", 0, 0, 22, 16), ("ellipse", -12, 5, 11, 15),
            ("circle", -10, -6, 7), ("line", -16, -2, -21, 12, 3)]),
        ("giraffe", 0.08, [
            ("ellipse", 0, 4, 16, 10), ("line", 2, -6, 2, -28, 5),
            ("ellipse", 2, -30, 8, 6), ("circle", -1, -14, 2), ("circle", 1, -22, 2)]),
        ("cat", 0.50, [
            ("ellipse", 0, 0, 16, 12), ("circle", 10, -6, 6),
            ("poly", [(7, -12), (10, -18), (13, -12)]), ("poly", [(8, -12), (10, -20), (12, -12)]),
            ("line", -10, 0, -18, 8, 3)]),
        ("dog", 0.06, [
            ("ellipse", 0, 0, 18, 10), ("ellipse", 12, -4, 9, 7),
            ("ellipse", 16, -2, 6, 8), ("line", -12, 0, -18, -10, 3)]),
        ("frog", 0.28, [
            ("circle", 0, 2, 10), ("circle", -5, -5, 3), ("circle", 5, -5, 3),
            ("ellipse", -8, 10, 4, 8), ("ellipse", 8, 10, 4, 8)]),
        ("bird", 0.55, [
            ("ellipse", 0, 0, 18, 10), ("ellipse", -4, -4, 10, 7),
            ("poly", [(10, 0), (16, -4), (10, 4)]), ("line", 14, 2, 18, 6, 2)]),
        ("fish", 0.70, [
            ("ellipse", 0, 0, 20, 9), ("poly", [(-10, 0), (-18, -7), (-18, 7)]),
            ("poly", [(4, -4), (10, -10), (6, -2)])]),
        ("butterfly", 0.85, [
            ("ellipse", -8, -4, 12, 16), ("ellipse", 8, -4, 12, 16),
            ("ellipse", -4, 6, 8, 10), ("ellipse", 4, 6, 8, 10),
            ("ellipse", 0, 0, 3, 10)]),
        ("bunny", 0.92, [
            ("ellipse", 0, 2, 14, 12), ("circle", 0, -8, 8),
            ("ellipse", -3, -20, 5, 14), ("ellipse", 3, -20, 5, 14)]),
        ("owl", 0.12, [
            ("circle", 0, 2, 12), ("circle", -4, -1, 4), ("circle", 4, -1, 4),
            ("circle", -4, -1, 2), ("circle", 4, -1, 2),
            ("poly", [(-6, -10), (-8, -16), (-4, -12)]), ("poly", [(6, -10), (8, -16), (4, -12)])]),
        ("bear", 0.04, [
            ("circle", 0, 4, 14), ("circle", 0, -8, 9),
            ("circle", -8, -14, 5), ("circle", 8, -14, 5)]),
        ("turtle", 0.30, [
            ("ellipse", 0, 0, 22, 14), ("circle", 14, 0, 5),
            ("ellipse", -8, 10, 5, 3), ("ellipse", 8, 10, 5, 3)]),
        ("penguin", 0.58, [
            ("ellipse", 0, 0, 16, 22), ("ellipse", 0, 4, 10, 12),
            ("poly", [(4, 10), (8, 14), (4, 14)]),
            ("ellipse", -6, -10, 4, 3), ("ellipse", 6, -10, 4, 3)]),
        ("fox", 0.10, [
            ("ellipse", 0, 2, 18, 12), ("poly", [(-10, -6), (-14, -14), (-6, -6)]),
            ("poly", [(10, -6), (14, -14), (6, -6)]), ("ellipse", -16, 0, 10, 6)]),
        ("monkey", 0.08, [
            ("circle", 0, 0, 10), ("circle", -8, -8, 6), ("circle", 8, -8, 6),
            ("circle", -3, -2, 2), ("circle", 3, -2, 2),
            ("line", -4, 8, -18, 18, 3)]),
    ]

    def __init__(self, rw, rh):
        self.rw, self.rh = rw, rh
        self.animals = []
        self._group_centers = []
        self._init_animals()

    def _init_animals(self):
        self.animals.clear()
        n = self.N
        nk = len(self.ANIMALS)
        margin = 80
        gw = (self.rw - margin * 2) / self.GROUPS
        gh = (self.rh - margin * 2) / max(1, self.GROUPS // 2)
        self._group_centers = []
        for g in range(self.GROUPS):
            gx = margin + (g % (self.GROUPS // 2 + 1)) * (gw + margin)
            gy = margin + (g // (self.GROUPS // 2 + 1)) * (gh + margin)
            self._group_centers.append([float(gx), float(gy)])

        for i in range(n):
            kind_idx = i % nk
            name, hue_base, shapes = self.ANIMALS[kind_idx]
            gid = i % self.GROUPS
            cx, cy = self._group_centers[gid]
            off_x = random.uniform(-40, 40)
            off_y = random.uniform(-30, 30)
            a = PartyAnimal(name, hue_base, shapes, cx + off_x, cy + off_y,
                            gid, hue_base + i * 0.07)
            self.animals.append(a)

    def step(self, dt, beat, t):
        for g in range(self.GROUPS):
            cx, cy = self._group_centers[g]
            drift = math.sin(t * 0.15 + g * 1.3) * 60
            drift2 = math.cos(t * 0.12 + g * 2.1) * 50
            self._group_centers[g] = [cx + drift * dt * 0.1,
                                      cy + drift2 * dt * 0.1]
            # keep in bounds
            cx, cy = self._group_centers[g]
            self._group_centers[g][0] = max(80, min(self.rw - 80, cx))
            self._group_centers[g][1] = max(60, min(self.rh - 60, cy))

        for a in self.animals:
            cx, cy = self._group_centers[a.group_id]
            a.step(dt, beat, t, cx, cy)

    def draw(self, surf, col_fn, beat, t, scale=1.0):
        for a in self.animals:
            a.draw(surf, col_fn, beat, t, scale)


class PartyAnimal:
    SIZE = 1.0   # base scale

    def __init__(self, kind, hue_base, shapes, x, y, group_id, hue):
        self.kind = kind
        self.hue_base = hue_base
        self.shapes = shapes
        self.x, self.y = x, y
        self.group_id = group_id
        self.hue = hue
        self.angle = random.random() * TAU * 0.5
        self.bob = random.random() * TAU
        self.flip = random.choice([-1, 1])

    def step(self, dt, beat, t, cx, cy):
        speed = (0.6 + beat * 1.2) * self.flip
        self.angle += dt * speed * 0.8
        self.bob += dt * (4 + beat * 3)
        dist = 25 + beat * 15
        dx = math.cos(self.angle) * dist
        dy = math.sin(self.bob * 0.7) * (12 + beat * 6)
        self.x = cx + dx
        self.y = cy + dy

    def draw(self, surf, col_fn, beat, t, scale=1.0):
        squeeze = 1.0 + beat * 0.15
        bx = self.x * scale
        by = self.y * scale
        for i, (cmd, *args) in enumerate(self.shapes):
            frac = (self.hue_base + i * 0.07 + math.sin(t * 0.5 + i) * 0.1) % 1.0
            v = 0.6 + 0.4 * (1 - abs(math.sin(self.bob + i)))
            c = col_fn(frac, v)
            if cmd == "circle":
                cx, cy, r = args
                px = bx + cx * scale
                py = by + cy * scale * squeeze
                pr = r * scale
                pygame.draw.circle(surf, c, (int(px), int(py)), max(1, int(pr)))
            elif cmd == "ellipse":
                cx, cy, rx, ry = args
                px = bx + cx * scale
                py = by + cy * scale * squeeze
                pw = rx * scale * 2
                ph = ry * scale * 2 * squeeze
                rect = pygame.Rect(int(px - pw / 2), int(py - ph / 2),
                                   max(2, int(pw)), max(2, int(ph)))
                pygame.draw.ellipse(surf, c, rect)
            elif cmd == "line":
                x1, y1, x2, y2, w = args
                p1 = (int(bx + x1 * scale), int(by + y1 * scale * squeeze))
                p2 = (int(bx + x2 * scale), int(by + y2 * scale * squeeze))
                pygame.draw.line(surf, c, p1, p2, max(1, int(w * scale)))
            elif cmd == "poly":
                pts = args[0]
                ppts = [(int(bx + x * scale), int(by + y * scale * squeeze))
                        for x, y in pts]
                if len(ppts) >= 3:
                    pygame.draw.polygon(surf, c, ppts)


# --------------------------------------------------------------------------
# presets: parameter bundles that reference a scene + motion
# --------------------------------------------------------------------------
def make_presets():
    # Per-preset "look" knobs, alongside motion:
    #   bloom  0..1  glow strength         flash  0..1  beat-flash strength
    #   emit   int   particles per beat    aa     bool   crisp anti-aliased lines
    #   shapes int   additive glow blobs
    # Loud presets keep bloom/flash/emit high; the "sharp" ones near zero.
    return [
        # --- crisp / poignant: minimal glow, no flash, defined lines ----------
        dict(name="Contour",    scene="circle",    zoom=1.002, zoom_beat=0.01,
             rot=0.04, rot_wob=0.15, decay=30, hue_speed=0.02, shapes=0, warp=0,
             bloom=0.18, flash=0.0, emit=0, aa=True),
        dict(name="Minimal",    scene="horizon",   zoom=1.000, zoom_beat=0.004,
             rot=0.0, rot_wob=0.04, decay=44, hue_speed=0.015, shapes=0, warp=0,
             bloom=0.12, flash=0.0, emit=0, aa=True),
        dict(name="Ink Scope",  scene="lissajous", zoom=1.000, zoom_beat=0.006,
             rot=0.06, rot_wob=0.2, decay=38, hue_speed=0.03, shapes=0, warp=0,
             bloom=0.14, flash=0.0, emit=0, aa=True),
        dict(name="Blueprint",  scene="grid",      zoom=1.002, zoom_beat=0.006,
             rot=0.0, rot_wob=0.05, decay=26, hue_speed=0.02, shapes=0, warp=0,
             bloom=0.16, flash=0.0, emit=0, aa=True),
        # --- balanced ---------------------------------------------------------
        dict(name="Equalizer",  scene="bars",      zoom=1.004, zoom_beat=0.03,
             rot=0.0, rot_wob=0.1, decay=12, hue_speed=0.04, shapes=0, warp=0,
             bloom=0.35, flash=0.1, emit=0, aa=False),
        dict(name="Aurora",     scene="dual",      zoom=1.006, zoom_beat=0.015,
             rot=0.05, rot_wob=0.3, decay=10, hue_speed=0.06, shapes=1, warp=2,
             bloom=0.45, flash=0.15, emit=20, aa=True),
        dict(name="Warp Stars", scene="starfield", zoom=1.008, zoom_beat=0.02,
             rot=0.02, rot_wob=0.2, decay=8, hue_speed=0.05, shapes=0, warp=0,
             bloom=0.4, flash=0.1, emit=0, aa=False),
        dict(name="Vortex",     scene="spokes",    zoom=1.012, zoom_beat=0.02,
             rot=0.45, rot_wob=1.2, decay=10, hue_speed=0.05, shapes=2, warp=0,
             bloom=0.5, flash=0.25, emit=40, aa=False),
        # --- loud / explosive -------------------------------------------------
        dict(name="Hyperspace", scene="circle",    zoom=1.028, zoom_beat=0.05,
             rot=0.10, rot_wob=0.6, decay=10, hue_speed=0.03, shapes=3, warp=0,
             bloom=0.7, flash=0.5, emit=80, aa=False),
        dict(name="Dutchman",   scene="horizon",   zoom=1.035, zoom_beat=0.09,
             rot=-0.18, rot_wob=2.2, decay=14, hue_speed=0.02, shapes=2, warp=8,
             bloom=0.8, flash=0.6, emit=100, aa=False),
        dict(name="Nova Pulse", scene="ring",      zoom=1.020, zoom_beat=0.12,
             rot=0.28, rot_wob=1.0, decay=11, hue_speed=0.10, shapes=5, warp=2,
             bloom=0.85, flash=0.7, emit=110, aa=False),
        dict(name="Supernova",  scene="particles", zoom=1.018, zoom_beat=0.08,
             rot=0.12, rot_wob=0.6, decay=12, hue_speed=0.09, shapes=0, warp=2,
             bloom=1.0, flash=0.85, emit=140, aa=False),
        dict(name="Jurassic",  scene="dino", zoom=1.0, zoom_beat=0.0,
             rot=0.0, rot_wob=0.0, decay=255, hue_speed=0.04, shapes=0, warp=0,
             bloom=0.0, flash=0.0, emit=0, aa=True),
        dict(name="Party", scene="party", zoom=1.0, zoom_beat=0.0,
             rot=0.0, rot_wob=0.0, decay=30, hue_speed=0.08, shapes=0, warp=0,
             bloom=0.2, flash=0.0, emit=0, aa=True),
    ]


def lerp(a, b, k):
    return a + (b - a) * k


def blend_preset(a, b, k):
    out = {}
    for key in a:
        va = a[key]
        if isinstance(va, (int, float)) and not isinstance(va, bool):
            out[key] = lerp(va, b[key], k)
        else:
            out[key] = va if k < 0.5 else b[key]
    return out


SCENES = ["circle", "horizon", "dual", "bars", "spokes",
          "lissajous", "ring", "starfield", "grid", "particles", "dino", "party"]
MIRRORS = ["none", "lr", "quad", "kaleido"]

# Speed range for the up/down keys.  Multiplicative stepping gives fine
# control down low, so 0.02 (a near-frozen ~50x slow crawl) is usable.
SPEED_MIN, SPEED_MAX = 0.02, 4.0


# --------------------------------------------------------------------------
# main visualizer
# --------------------------------------------------------------------------
class Milky:
    def __init__(self, args):
        pygame.init()
        pygame.display.set_caption("~ MILKY :: milkdrop mode ~")
        self.RW, self.RH = args.width, args.height
        self.CX, self.CY = self.RW / 2, self.RH / 2
        self.fullscreen = args.fullscreen
        self._vsync = not args.no_vsync
        self._max_fps = args.max_fps
        self._base_flags = (pygame.OPENGL | pygame.DOUBLEBUF) if args.gpu else 0
        flags = self._base_flags | (
            pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE)
        self.screen = pygame.display.set_mode(
            (0, 0) if self.fullscreen else (self.RW, self.RH), flags,
            vsync=self._vsync if args.gpu else 0)
        self.win_size = self.screen.get_size()

        self._gpu_enabled = args.gpu
        self.gpu = None
        if args.gpu:
            import mgpu
            self.gpu = mgpu.GPURenderer(self.RW, self.RH)

        self.canvas = pygame.Surface((self.RW, self.RH)).convert()
        self.canvas.fill((0, 0, 0))
        self.tmp = pygame.Surface((self.RW, self.RH)).convert()

        self.audio = AudioEngine(live=args.audio)
        self.pal = Palette()
        self.presets = make_presets()
        self.stars = Stars(500, self.RW, self.RH)
        self.parts = Particles()
        self.dino = DinoGame(self.RW, self.RH)
        self.dino_on = False
        self.party = PartyScene(self.RW, self.RH)

        # start on requested preset if any
        start = 0
        if args.preset:
            for i, pr in enumerate(self.presets):
                if pr["name"].lower().startswith(args.preset.lower()):
                    start = i
                    break
        self.cur_i = start
        self.next_i = (start + 1) % len(self.presets)
        self.morph = 1.0
        self.preset_time = 0.0
        self.preset_len = 16.0

        self.scene_override = None       # forced scene via n/p keys
        self.mirror = 0
        self.kaleido = False
        self.use_bloom = not args.nobloom
        self.beat_flash = True
        self.auto = True
        self.sharp = args.sharp          # global "calm/defined" override
        self._aa = False                 # current preset anti-alias flag
        self.decay_bias = 0
        self.zoom_bias = 0.0

        self.t = 0.0
        self.dt = 1 / 60.0               # last frame delta (speed-scaled)
        self.speed = min(SPEED_MAX, max(SPEED_MIN, args.speed))
        self.hue = 0.0
        self.show_hud = True
        self.show_perf = False
        self._fps_history = deque(maxlen=300)
        self._benchmark_frames = args.benchmark
        self.font = pygame.font.SysFont("consolas", 15, bold=True)
        self.big = pygame.font.SysFont("consolas", 22, bold=True)
        self.clock = pygame.time.Clock()
        self.fps = 0.0

    # ---- preset control ----
    def cur_params(self):
        a = self.presets[self.cur_i]
        b = self.presets[self.next_i]
        k = min(1.0, self.morph)
        k = k * k * (3 - 2 * k)
        p = blend_preset(a, b, k)
        name = b["name"] if k >= 0.5 else a["name"]
        if self.scene_override:
            p["scene"] = self.scene_override
        return p, name

    def go_next(self, idx=None):
        self.cur_i = self.next_i
        self.next_i = idx if idx is not None else (self.next_i + 1) % len(self.presets)
        self.morph = 0.0
        self.preset_time = 0.0
        self.scene_override = None

    # ---- feedback warp (the core milkdrop trick) ----
    def feedback(self, p, dt):
        beat = self.audio.beat
        zoom = p["zoom"] + p["zoom_beat"] * beat + self.zoom_bias
        rot = (p["rot"] + p["rot_wob"] * math.sin(self.t * 0.7)) * (1 + beat)
        warped = pygame.transform.rotozoom(self.canvas, rot * dt * 60, zoom)
        rect = warped.get_rect(center=(self.CX, self.CY))
        if p["warp"] > 0.5:
            rect.x += int(math.sin(self.t * 1.3) * p["warp"] * (0.5 + beat))
            rect.y += int(math.cos(self.t * 1.1) * p["warp"] * (0.5 + beat))
        self.canvas.fill((0, 0, 0))
        self.canvas.blit(warped, rect)
        d = max(1, int(p["decay"]) + self.decay_bias)
        self.canvas.fill((d, d, d), special_flags=pygame.BLEND_RGB_SUB)

    # ---- scenes ----
    def _col(self, frac, v=1.0):
        return self.pal.color(frac, self.hue, v)

    def scene_circle(self, p, wave, spec):
        beat = self.audio.beat
        r0 = min(self.RW, self.RH) * (0.20 + 0.06 * beat)
        amp = 120 * (0.5 + beat)
        pts = []
        for i in range(0, len(wave), 2):
            a = i / len(wave) * TAU
            rr = r0 + wave[i] * amp
            pts.append((self.CX + rr * math.cos(a), self.CY + rr * math.sin(a)))
        pts.append(pts[0])
        self.polyline(self._col(0.1), pts, False, 2)

    def scene_horizon(self, p, wave, spec):
        beat = self.audio.beat
        amp = 160 * (0.5 + beat)
        step = self.RW / (len(wave) - 1)
        pts = [(i * step, self.CY + wave[i] * amp) for i in range(len(wave))]
        self.polyline(self._col(0.0), pts, False, 2)
        pts2 = [(i * step, self.CY * 0.5 + wave[i] * amp * 0.6) for i in range(len(wave))]
        self.polyline(self._col(0.5), pts2, False, 1)

    def scene_dual(self, p, wave, spec):
        beat = self.audio.beat
        amp = 130 * (0.5 + beat)
        step = self.RW / (len(wave) - 1)
        for c in (self._col(0.0), self._col(0.5)):
            pts = [(i * step, self.CY + wave[i] * amp * math.sin(i / len(wave) * math.pi))
                   for i in range(len(wave))]
            self.polyline(c, pts, False, 2)
        r0 = min(self.RW, self.RH) * 0.16
        pts = [(self.CX + (r0 + wave[i] * 90 * (0.5 + beat)) * math.cos(i / len(wave) * TAU),
                self.CY + (r0 + wave[i] * 90 * (0.5 + beat)) * math.sin(i / len(wave) * TAU))
               for i in range(0, len(wave), 2)]
        self.polyline(self._col(0.25), pts, True, 2)

    def scene_bars(self, p, wave, spec):
        n = len(spec)
        bw = self.RW / n
        for i, v in enumerate(spec):
            h = float(v) * self.RH * 0.9
            c = self._col(i / n)
            x = i * bw
            pygame.draw.rect(self.canvas, c, (x + 1, self.RH - h, bw - 2, h))
            # mirror on top for symmetry
            pygame.draw.rect(self.canvas, self._col(i / n, 0.4), (x + 1, 0, bw - 2, h * 0.4))

    def scene_spokes(self, p, wave, spec):
        n = len(spec)
        for i, v in enumerate(spec):
            a = i / n * TAU
            r = min(self.RW, self.RH) * 0.1 + float(v) * min(self.RW, self.RH) * 0.45
            x = self.CX + r * math.cos(a + self.t * 0.3)
            y = self.CY + r * math.sin(a + self.t * 0.3)
            pygame.draw.line(self.canvas, self._col(i / n), (self.CX, self.CY), (x, y), 2)

    def scene_lissajous(self, p, wave, spec):
        beat = self.audio.beat
        a = 3 + 2 * math.sin(self.t * 0.2)
        b = 2 + 2 * math.cos(self.t * 0.17)
        R = min(self.RW, self.RH) * 0.4 * (0.7 + 0.3 * beat)
        pts = []
        m = 400
        for i in range(m):
            u = i / m * TAU
            x = self.CX + R * math.sin(a * u + self.t)
            y = self.CY + R * math.sin(b * u)
            pts.append((x, y))
        self.polyline(self._col(0.3), pts, True, 2)

    def scene_ring(self, p, wave, spec):
        n = len(spec)
        r0 = min(self.RW, self.RH) * 0.22
        for i, v in enumerate(spec):
            a = i / n * TAU + self.t * 0.2
            r1 = r0 + float(v) * 160
            x0, y0 = self.CX + r0 * math.cos(a), self.CY + r0 * math.sin(a)
            x1, y1 = self.CX + r1 * math.cos(a), self.CY + r1 * math.sin(a)
            pygame.draw.line(self.canvas, self._col(i / n), (x0, y0), (x1, y1), 3)

    def scene_starfield(self, p, wave, spec):
        beat = self.audio.beat
        self.stars.step(0.006 + 0.02 * beat)
        px = self.stars.p
        sx = self.CX + (px[:, 0] / px[:, 2]) * self.CX
        sy = self.CY + (px[:, 1] / px[:, 2]) * self.CY
        size = ((1 - px[:, 2]) * 4).clip(1, 5)
        for i in range(len(px)):
            if 0 <= sx[i] < self.RW and 0 <= sy[i] < self.RH:
                c = self._col(float(px[i, 2]))
                s = int(size[i])
                pygame.draw.circle(self.canvas, c, (int(sx[i]), int(sy[i])), s)

    def scene_grid(self, p, wave, spec):
        beat = self.audio.beat
        horizon = self.CY
        off = (self.t * 0.5) % 1.0
        # perspective floor lines
        aa = self._aa
        for i in range(1, 16):
            z = (i - off) / 16
            y = horizon + (self.RH - horizon) * (z ** 1.6)
            c = self._col(i / 16, 0.6 + 0.4 * beat)
            (pygame.draw.aaline if aa else pygame.draw.line)(
                self.canvas, c, (0, y), (self.RW, y))
        for j in range(-8, 9):
            x0 = self.CX + j * 14
            x1 = self.CX + j * self.RW / 8
            (pygame.draw.aaline if aa else pygame.draw.line)(
                self.canvas, self._col(0.5), (x0, horizon), (x1, self.RH))

    def scene_particles(self, p, wave, spec):
        # emission handled in draw_overlay via beat; just render here
        self.parts.step(self.dt)          # speed-scaled so slow-mo slows drift too
        for i in range(len(self.parts.pos)):
            x, y = self.parts.pos[i]
            if 0 <= x < self.RW and 0 <= y < self.RH:
                life = float(self.parts.life[i])
                c = hsv2rgb(float(self.parts.hue[i]) + self.hue,
                            self.pal.cur()["sat"], life)
                r = max(1, int(life * 4))
                pygame.draw.circle(self.canvas, c, (int(x), int(y)), r)

    def scene_dino(self, p, wave, spec):
        beat = self.audio.beat
        self.dino.spd = max(self.dino.spd, 2.0 + beat * 4)
        self.dino.step(self.audio.hit, self.dt)
        # drawing moved to _draw_scene_post (full window resolution, crisp)

    def scene_party(self, p, wave, spec):
        # ambient background — animals drawn at full res via _draw_scene_post
        pass

    SCENE_FN = {
        "circle": scene_circle, "horizon": scene_horizon, "dual": scene_dual,
        "bars": scene_bars, "spokes": scene_spokes, "lissajous": scene_lissajous,
        "ring": scene_ring, "starfield": scene_starfield, "grid": scene_grid,
        "particles": scene_particles, "dino": scene_dino, "party": scene_party,
    }

    def polyline(self, color, pts, closed=False, width=2):
        """Crisp anti-aliased line when the preset/sharp-mode asks for it,
        otherwise a solid width line."""
        if len(pts) < 2:
            return
        if self._aa:
            pygame.draw.aalines(self.canvas, color, closed, pts)
        else:
            pygame.draw.lines(self.canvas, color, closed, pts, width)

    def draw_scene(self, p, wave, spec):
        wave = [float(v) for v in wave]   # pygame wants plain-float coord pairs
        self._aa = bool(p.get("aa")) or self.sharp
        self.SCENE_FN[p["scene"]](self, p, wave, spec)

    # ---- overlays shared by all scenes ----
    def draw_overlay(self, p, wave, spec):
        beat = self.audio.beat
        sharp = self.sharp

        # (dino overlay now drawn after bloom to avoid feedback trail)
        # pulsing additive glow shapes (skipped entirely in sharp mode)
        nshapes = 0 if sharp else int(round(p["shapes"]))
        for s in range(nshapes):
            ang = self.t * (0.4 + 0.2 * s) + s * TAU / max(1, nshapes)
            dist = (80 + 40 * s) * (0.6 + beat)
            sx = self.CX + math.cos(ang) * dist
            sy = self.CY + math.sin(ang) * dist
            rad = int(18 + 26 * beat + 6 * s)
            c = self._col(0.13 * s)
            glow = pygame.Surface((rad * 2, rad * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*c, 200), (rad, rad), rad)
            self.canvas.blit(glow, (sx - rad, sy - rad),
                             special_flags=pygame.BLEND_RGB_ADD)

        # particle emission is now per-preset, not on every scene.
        emit = int(p.get("emit", 0))
        if p["scene"] == "particles":
            emit = max(emit, 40)          # keep the particles scene alive
        if sharp:
            emit //= 3
        flash = 0.0 if sharp else float(p.get("flash", 0.0))

        if self.audio.hit:
            if emit > 0:
                self.parts.burst(self.CX, self.CY, emit, 6 + 10 * beat, random.random())
            if self.beat_flash and flash > 0:
                fv = int((14 + 30 * beat) * flash)
                if fv > 0:
                    fl = pygame.Surface((self.RW, self.RH))
                    fl.fill((fv, fv, fv))
                    self.canvas.blit(fl, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    # ---- dino overlay (rendered after bloom to avoid feedback trails) ----
    def _draw_dino_overlay(self, p, wave, spec, *, target=None):
        """Draw the dino overlay on *target* (or self.tmp), returning the surface.
        For the CPU path this is called after bloom; for GPU we produce RGBA bytes."""
        if not (self.dino_on and p.get("scene") != "dino"):
            return None
        surf = target if target is not None else self.tmp
        sx = surf.get_width()
        scale = sx / self.RW if sx != self.RW else 1.0
        self.dino.step(self.audio.hit, self.dt)
        self.dino.draw(surf, self._col, scale)
        return surf

    # ---- scene post-draw (full-resolution scene content, after smoothscale) ----
    def _draw_scene_post(self, p, wave, spec, *, target=None):
        """Draw scene-specific content at full window resolution (after the
        feedback/bloom/smoothscale pipeline).  Returns the target if drawn."""
        if target is None:
            target = self.tmp
        scene = p["scene"]
        if scene in ("party", "dino"):
            sx = target.get_width()
            scale = sx / self.RW if sx != self.RW else 1.0
            if scene == "party":
                self.party.draw(target, self._col, self.audio.beat, self.t, scale)
            else:
                self.dino.draw(target, self._col, scale)
            return target
        return None

    # ---- mirror / kaleidoscope post FX ----
    def apply_mirror(self):
        mode = MIRRORS[self.mirror]
        if mode == "none" and not self.kaleido:
            return
        if mode == "lr":
            half = self.canvas.subsurface((0, 0, self.RW // 2, self.RH)).copy()
            self.canvas.blit(pygame.transform.flip(half, True, False),
                             (self.RW // 2, 0))
        elif mode == "quad":
            q = self.canvas.subsurface((0, 0, self.RW // 2, self.RH // 2)).copy()
            self.canvas.blit(pygame.transform.flip(q, True, False), (self.RW // 2, 0))
            self.canvas.blit(pygame.transform.flip(q, False, True), (0, self.RH // 2))
            self.canvas.blit(pygame.transform.flip(q, True, True),
                             (self.RW // 2, self.RH // 2))
        elif mode == "kaleido" or self.kaleido:
            base = self.canvas.copy()
            for a in (60, 120, 180, 240, 300):
                rot = pygame.transform.rotozoom(base, a, 1.0)
                rect = rot.get_rect(center=(self.CX, self.CY))
                self.canvas.blit(rot, rect, special_flags=pygame.BLEND_RGB_ADD)
            self.canvas.blit(pygame.transform.flip(base, True, False), (0, 0),
                             special_flags=pygame.BLEND_RGB_ADD)

    # ---- bloom ----
    def bloom(self, target, strength):
        target.blit(self.canvas, (0, 0))
        if not self.use_bloom or strength <= 0.01:
            return
        strength = min(1.0, strength)
        small = pygame.transform.smoothscale(self.canvas, (self.RW // 5, self.RH // 5))
        blur = pygame.transform.smoothscale(small, (self.RW, self.RH))
        if strength < 1.0:                      # attenuate the glow before adding
            k = int(255 * strength)
            blur.fill((k, k, k), special_flags=pygame.BLEND_RGB_MULT)
        target.blit(blur, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    # ---- HUD ----
    def draw_hud(self, name, p, surf):
        if not self.show_hud:
            return
        sw, sh_ = surf.get_width(), surf.get_height()
        beat_bar = "#" * int(self.audio.beat * 20)
        src = "MIC" if self.audio.live else "synth"
        vsync_str = f"vsync {'on' if self._vsync else 'off'}"
        lines = [
            (self.big, f"{name}  ::  {p['scene']}",
             self._col(0.3, 1.0)),
            (self.font, f"palette {self.pal.cur()['name']}   mirror {MIRRORS[self.mirror]}"
                        f"{'+kaleido' if self.kaleido else ''}   "
                        f"bloom {'on' if self.use_bloom else 'off'}   "
                        f"sharp {'ON' if self.sharp else 'off'}   "
                        f"{vsync_str}   "
                        f"speed {self.speed:0.2f}x   "
                        f"auto {'on' if self.auto else 'off'}   {int(self.fps)}fps  [{src}]",
             (200, 220, 230)),
            (self.font, f"beat {beat_bar:<20}", self._col(0.6)),
            (self.font, "space next  n/p scene  c palette  m mirror  k kaleido  "
                        "x sharp  v vsync  g perf  up/down speed  0 reset  s shot  h hud  f full  q quit",
             (150, 165, 180)),
        ]
        y = 8
        for f, txt, c in lines:
            surf.blit(f.render(txt, True, (0, 0, 0)), (13, y + 1))
            surf.blit(f.render(txt, True, c), (12, y))
            y += f.get_height() + 4

        # performance stats overlay
        if self.show_perf and len(self._fps_history) > 10:
            arr = list(self._fps_history)
            avg = sum(arr) / len(arr)
            mn = min(arr)
            mx = max(arr)
            p99 = sorted(arr)[int(len(arr) * 0.99)]
            gpu_info = "GPU" if self.gpu else "CPU"
            lines2 = [
                (self.font, f"[{gpu_info}]  FPS  min {mn:.0f}  avg {avg:.0f}  max {mx:.0f}  p99 {p99:.0f}",
                 (100, 220, 100)),
            ]
            for f, txt, c in lines2:
                surf.blit(f.render(txt, True, (0, 0, 0)), (13, y + 1))
                surf.blit(f.render(txt, True, c), (12, y))
                y += f.get_height() + 4

            # mini FPS sparkline
            spw, sph = sw - 24, 32
            spx, spy = 12, max(0, sh_ - sph - 12)
            spark = pygame.Surface((spw, sph), pygame.SRCALPHA)
            spark.fill((0, 0, 0, 100))
            n = min(len(arr), spw)
            arr_n = arr[-n:]
            mx_val = max(arr_n) or 1
            for i in range(1, n):
                x0 = (i - 1) * spw / n
                x1 = i * spw / n
                y0 = sph - (arr_n[i - 1] / mx_val) * (sph - 2) - 1
                y1 = sph - (arr_n[i] / mx_val) * (sph - 2) - 1
                c = (100, 220, 100) if arr_n[i] >= 55 else (240, 180, 40)
                pygame.draw.line(spark, c, (x0, y0), (x1, y1), 2)
                if arr_n[i] < 30:
                    pygame.draw.circle(spark, (240, 60, 60), (int(x1), int(y1)), 3)
            surf.blit(spark, (spx, spy))

    # ---- GPU render path ----
    def _reinit_gpu(self):
        import mgpu
        self.gpu = mgpu.GPURenderer(self.RW, self.RH)

    def _hud_bytes(self, name, p):
        if not self.show_hud:
            return None
        surf = pygame.Surface(self.win_size, pygame.SRCALPHA)
        self.draw_hud(name, p, surf)
        return pygame.image.tobytes(surf, "RGBA")

    def render_gpu(self, p, name, wave, spec):
        beat = self.audio.beat
        zoom = p["zoom"] + p["zoom_beat"] * beat + self.zoom_bias
        rot = (p["rot"] + p["rot_wob"] * math.sin(self.t * 0.7)) * (1 + beat)
        angle = math.radians(rot * self.dt * 60)
        ox = oy = 0.0
        if p["warp"] > 0.5:
            ox = math.sin(self.t * 1.3) * p["warp"] * (0.5 + beat) / self.RW
            oy = math.cos(self.t * 1.1) * p["warp"] * (0.5 + beat) / self.RH
        d = max(1, int(p["decay"]) + self.decay_bias)
        decay = max(0.0, 1.0 - d / 255.0)

        # scene layer on a fresh black canvas -- accumulation lives on the GPU
        self.canvas.fill((0, 0, 0))
        self.draw_scene(p, wave, spec)
        self.draw_overlay(p, wave, spec)

        # dino overlay as RGBA bytes (drawn on a separate transparent surface)
        dino_bytes = None
        if self.dino_on and p.get("scene") != "dino":
            dino_surf = pygame.Surface(self.win_size, pygame.SRCALPHA)
            self._draw_dino_overlay(p, wave, spec, target=dino_surf)
            dino_bytes = pygame.image.tobytes(dino_surf, "RGBA")

        # scene post layer (scene content drawn at full win_size resolution)
        scene_post_bytes = None
        if p.get("scene") in ("party", "dino"):
            if p.get("scene") == "party":
                self.party.step(self.dt, self.audio.beat, self.t)
            post_surf = pygame.Surface(self.win_size, pygame.SRCALPHA)
            self._draw_scene_post(p, wave, spec, target=post_surf)
            scene_post_bytes = pygame.image.tobytes(post_surf, "RGBA")

        bloom_amt = 0.0
        if self.use_bloom:
            bloom_amt = float(p.get("bloom", 0.6)) * (0.3 if self.sharp else 1.0)
        mirror_id = 3 if self.kaleido else self.mirror   # MIRRORS index == shader id

        self.gpu.render(
            angle=angle, zoom=zoom, offset=(ox, oy), decay=decay,
            scene_bytes=pygame.image.tobytes(self.canvas, "RGB"),
            mirror_id=mirror_id, bloom_amt=bloom_amt,
            hud_bytes=self._hud_bytes(name, p),
            dino_bytes=dino_bytes,
            scene_post_bytes=scene_post_bytes,
            win_size=self.win_size)

    # ---- screenshot ----
    def _save_screen(self, path):
        if self.gpu:
            w, h = self.gpu.ctx.screen.size
            data = self.gpu.ctx.screen.read(components=3)
            surf = pygame.image.frombytes(data, (w, h), "RGB", True)
        else:
            surf = self.screen
        pygame.image.save(surf, path)

    def screenshot(self):
        os.makedirs("shots", exist_ok=True)
        path = os.path.join("shots", time.strftime("milky_%Y%m%d_%H%M%S.png"))
        self._save_screen(path)
        print("saved", path)

    # ---- events ----
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.VIDEORESIZE and not self.fullscreen:
                self.win_size = (max(320, e.w), max(240, e.h))
                self.screen = pygame.display.set_mode(
                    self.win_size, pygame.RESIZABLE | self._base_flags,
                    vsync=self._vsync if self._gpu_enabled else 0)
                if self._gpu_enabled:
                    self.gpu = None
                    self._reinit_gpu()
            if e.type == pygame.KEYDOWN:
                k = e.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif k == pygame.K_SPACE:
                    self.go_next()
                elif k == pygame.K_r:
                    self.go_next(random.randrange(len(self.presets)))
                elif k == pygame.K_n:
                    self._cycle_scene(+1)
                elif k == pygame.K_p:
                    self._cycle_scene(-1)
                elif k == pygame.K_c:
                    self.pal.next()
                elif k == pygame.K_m:
                    self.mirror = (self.mirror + 1) % len(MIRRORS)
                elif k == pygame.K_k:
                    self.kaleido = not self.kaleido
                elif k == pygame.K_b:
                    self.use_bloom = not self.use_bloom
                elif k == pygame.K_x:
                    self.sharp = not self.sharp
                elif k == pygame.K_t:
                    self.beat_flash = not self.beat_flash
                elif k == pygame.K_a:
                    self.auto = not self.auto
                elif k == pygame.K_s:
                    self.screenshot()
                elif k == pygame.K_d:
                    self.dino_on = not self.dino_on
                    if self.dino_on:
                        self.dino.reset()
                elif k == pygame.K_g:
                    self.show_perf = not self.show_perf
                elif k == pygame.K_v:
                    self._vsync = not self._vsync
                    if self.gpu:
                        try:
                            pygame.display.set_vsync(self._vsync)
                        except Exception:
                            self._vsync = not self._vsync
                elif k == pygame.K_h:
                    self.show_hud = not self.show_hud
                elif k == pygame.K_UP:
                    self.speed = min(SPEED_MAX, self.speed * 1.18)
                elif k == pygame.K_DOWN:
                    self.speed = max(SPEED_MIN, self.speed / 1.18)
                elif k in (pygame.K_0, pygame.K_KP0):
                    self.speed = 1.0
                elif k == pygame.K_LEFTBRACKET:
                    self.decay_bias -= 1
                elif k == pygame.K_RIGHTBRACKET:
                    self.decay_bias += 1
                elif k == pygame.K_MINUS:
                    self.zoom_bias -= 0.004
                elif k in (pygame.K_EQUALS, pygame.K_PLUS):
                    self.zoom_bias += 0.004
                elif k == pygame.K_f:
                    self.toggle_fullscreen()
        return True

    def _cycle_scene(self, d):
        cur = self.scene_override or self.presets[self.cur_i]["scene"]
        i = (SCENES.index(cur) + d) % len(SCENES)
        self.scene_override = SCENES[i]

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        pygame.display.toggle_fullscreen()
        self.win_size = pygame.display.get_window_size()

    # ---- main loop ----
    def run(self, selftest_frames=None, shot_path=None):
        running = True
        frames = 0
        benchmark = self._benchmark_frames > 0
        bench_results = []
        bench_preset_frames = 0
        bench_preset_idx = 0
        bench_presets_run = []

        while running:
            tick_target = self._max_fps if self._max_fps > 0 else 0
            dt_ms = self.clock.tick(tick_target)
            self.fps = self.clock.get_fps()
            self._fps_history.append(self.fps)
            dt = min(0.05, dt_ms / 1000.0) * self.speed
            self.dt = dt
            self.t += dt

            if selftest_frames is None and not benchmark:
                running = self.handle_events()

            self.morph = min(1.0, self.morph + dt / 2.5)
            self.preset_time += dt
            if self.auto and self.preset_time > self.preset_len:
                self.go_next()
            p, name = self.cur_params()

            if benchmark:
                if bench_preset_idx >= len(self.presets):
                    break
                bench_preset_frames += 1
                bench_results.append(self.fps)
                if bench_preset_frames >= self._benchmark_frames:
                    avg = sum(bench_results[-self._benchmark_frames:]) / self._benchmark_frames
                    mn = min(bench_results[-self._benchmark_frames:])
                    mx = max(bench_results[-self._benchmark_frames:])
                    preset_name = self.presets[bench_preset_idx]["name"]
                    bench_presets_run.append((preset_name, mn, avg, mx))
                    bench_preset_idx += 1
                    bench_preset_frames = 0
                    if bench_preset_idx < len(self.presets):
                        self.go_next(bench_preset_idx)
                        self.auto = False

            wave, spec = self.audio.sample(self.t, max(1e-3, dt))
            self.hue += p["hue_speed"] * dt * 3.0
            if p.get("scene") == "party":
                self.party.step(self.dt, self.audio.beat, self.t)

            if self.gpu:
                self.render_gpu(p, name, wave, spec)
            else:
                self.feedback(p, dt)
                self.draw_scene(p, wave, spec)
                self.draw_overlay(p, wave, spec)
                self.apply_mirror()
                bloom_strength = float(p.get("bloom", 0.6)) * (0.3 if self.sharp else 1.0)
                self.bloom(self.tmp, bloom_strength)
                if self.win_size != (self.RW, self.RH):
                    pygame.transform.smoothscale(self.tmp, self.win_size, self.screen)
                else:
                    self.screen.blit(self.tmp, (0, 0))
                self._draw_dino_overlay(p, wave, spec, target=self.screen)
                self._draw_scene_post(p, wave, spec, target=self.screen)
                self.draw_hud(name, p, self.screen)
            frames += 1
            last = selftest_frames is not None and frames >= selftest_frames
            if last and shot_path:
                self._save_screen(shot_path)   # before flip: back buffer still holds it
            pygame.display.flip()
            if last:
                break

        self.audio.close()
        if benchmark and bench_presets_run:
            print("Benchmark results (FPS):")
            print(f"  {'Preset':<14} {'Min':>8} {'Avg':>8} {'Max':>8}")
            print("  " + "-" * 42)
            for pname, mn, avg, mx in bench_presets_run:
                print(f"  {pname:<14} {mn:>8.0f} {avg:>8.0f} {mx:>8.0f}")
            all_avgs = [r[2] for r in bench_presets_run]
            if all_avgs:
                print(f"  {'--- overall ---':<14} {'---':>8} {sum(all_avgs)/len(all_avgs):>8.0f} {'---':>8}")
        pygame.quit()


# --------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Milky - a Milkdrop-style visualizer")
    ap.add_argument("-W", "--width", type=int, default=960)
    ap.add_argument("-H", "--height", type=int, default=600)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--audio", action="store_true",
                    help="react to live microphone input (needs sounddevice)")
    ap.add_argument("--preset", default=None, help="start on a preset by name")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="initial speed (0.02 = near-frozen slow, 4.0 = fast)")
    ap.add_argument("--nobloom", action="store_true", help="disable the bloom pass")
    ap.add_argument("--gpu", action="store_true",
                    help="GPU render path (moderngl): warp/bloom/mirror on the GPU")
    ap.add_argument("--no-vsync", action="store_true",
                    help="disable vertical sync (GPU path only; default is vsync on)")
    ap.add_argument("--max-fps", type=int, default=0,
                    help="cap frame rate at N fps (0 = uncapped, default 0)")
    ap.add_argument("--sharp", action="store_true",
                    help="start in sharp mode: crisp lines, no flash, minimal glow")
    ap.add_argument("--list", action="store_true", help="list presets and exit")
    ap.add_argument("--benchmark", type=int, default=0, metavar="N",
                    help="run N frames through all presets and print FPS stats")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    if args.list:
        print("Presets:")
        for pr in make_presets():
            print(f"  {pr['name']:<12} scene={pr['scene']}")
        print("\nScenes:", ", ".join(SCENES))
        print("Palettes:", ", ".join(p["name"] for p in PALETTES))
        return

    if os.environ.get("CRAZY_SELFTEST") or os.environ.get("MILKY_SELFTEST"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        n = int(os.environ.get("MILKY_SELFTEST", os.environ.get("CRAZY_SELFTEST", "60")))
        m = Milky(args)
        # exercise every scene + mirror in the self-test
        m.run(selftest_frames=n)
        print("selftest OK")
        return

    if args.benchmark:
        m = Milky(args)
        m.run()
        return

    Milky(args).run()


if __name__ == "__main__":
    main()

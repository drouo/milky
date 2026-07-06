# Milky 🌌

A **Milkdrop-style music visualizer** in pure Python (pygame + numpy), with
optional live-audio input. It recreates the classic Winamp/Milkdrop feel — a
warping feedback tunnel, reactive waveforms, bloom glow, kaleidoscopic mirrors
and auto-morphing presets — and piles a bunch of extra scenes and live controls
on top.

![scenes](shots/) <!-- drop a screenshot here (press `s` in-app) -->

## Features

- **10 scenes** — `circle`, `horizon`, `dual`, `bars` (spectrum), `spokes`,
  `lissajous`, `ring`, `starfield`, `grid` (perspective tunnel), `particles`.
- **Feedback warp tunnel** — the core Milkdrop trick: every frame zooms +
  rotates the previous frame and blits it back, so everything trails into an
  endless morphing tunnel. Zoom/rotation react to the beat.
- **12 auto-morphing presets** that smoothly cross-fade one into the next —
  tuned across a range from crisp/poignant to loud/explosive.
- **Sharp mode** (`x` or `--sharp`) — dials any preset down to defined,
  anti-aliased lines on near-black: no beat flash, minimal glow, no particle
  spray. Presets **Contour / Minimal / Ink Scope / Blueprint** ship this way.
- **7 color palettes** — Rainbow, Fire, Ice, Toxic, Sunset, Neon, Mono.
- **Mirror / kaleidoscope** modes — none, left-right, quad, 6-fold kaleido.
- **Bloom glow** pass, **beat flash**, pulsing additive shapes, particle bursts.
- **Live audio** (optional) — reacts to your microphone via `sounddevice`, with
  FFT spectrum bars. Falls back to a built-in synthetic beat engine.
- **Screenshots**, fullscreen, resizable window, live speed/zoom/decay tweaks.

## Install

```bash
pip install pygame numpy
pip install sounddevice        # optional, for --audio
pip install moderngl           # optional, for --gpu
```

## Run

```bash
python milky.py                      # synthetic beat engine, windowed
python milky.py --audio              # react to your microphone
python milky.py --fullscreen
python milky.py -W 1280 -H 720
python milky.py --preset Supernova   # start on a named preset
python milky.py --sharp              # crisp/defined look, calm the flash & glow
python milky.py --list               # list presets/scenes/palettes and exit
python milky.py --nobloom            # disable bloom (faster on weak GPUs)
python milky.py --gpu                 # GPU render path (moderngl) -- much faster
```

### GPU render path

`--gpu` moves the feedback warp, bloom and mirror/kaleido onto the GPU as
fragment-shader passes (via `moderngl`); scenes are still drawn on the CPU and
uploaded once per frame. On a mid-range card it's roughly **4× faster** at 720p
and holds 60+ fps at 1080p, versus the pure-CPU path. Falls back to nothing —
if `moderngl` isn't installed, just omit the flag and the CPU renderer is used.

## Controls

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `Esc` / `q` | quit | `space` | next preset |
| `r` | random preset | `a` | toggle auto-cycle |
| `n` / `p` | next / prev scene | `c` | cycle palette |
| `m` | cycle mirror mode | `k` | toggle kaleidoscope |
| `x` | toggle sharp mode | `b` | toggle bloom |
| `t` | toggle beat flash | | |
| `[` / `]` | decay down / up | `-` / `=` | zoom down / up |
| `↑` / `↓` | speed up / down (0.02×–4×) | `0` | reset speed to 1× |
| `s` | save screenshot (PNG → `shots/`) | `h` | hide / show HUD |
| `f` | toggle fullscreen | | |

Speed steps multiplicatively, so you get fine control at the slow end — hold
`↓` and it eases all the way down to a near-frozen **0.02×** crawl (great for
the sharp presets). `0` snaps back to 1×. Start slow with `--speed 0.05`.

## How the live audio works

With `--audio`, a `sounddevice` input stream fills a rolling sample buffer; each
frame Milky takes an FFT to build the spectrum bars and low/mid/high band
energies, and does simple energy-based beat detection. Without it (or if
`sounddevice` isn't installed), a synthetic 125 BPM engine with fills drives the
visuals so it always looks alive.

## License

MIT — see [LICENSE](LICENSE).

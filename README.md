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
- **10 auto-morphing presets** that smoothly cross-fade one into the next.
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
```

## Run

```bash
python milky.py                      # synthetic beat engine, windowed
python milky.py --audio              # react to your microphone
python milky.py --fullscreen
python milky.py -W 1280 -H 720
python milky.py --preset Supernova   # start on a named preset
python milky.py --list               # list presets/scenes/palettes and exit
python milky.py --nobloom            # disable bloom (faster on weak GPUs)
```

## Controls

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `Esc` / `q` | quit | `space` | next preset |
| `r` | random preset | `a` | toggle auto-cycle |
| `n` / `p` | next / prev scene | `c` | cycle palette |
| `m` | cycle mirror mode | `k` | toggle kaleidoscope |
| `b` | toggle bloom | `t` | toggle beat flash |
| `[` / `]` | decay down / up | `-` / `=` | zoom down / up |
| `↑` / `↓` | speed up / down | `s` | save screenshot (PNG → `shots/`) |
| `h` | hide / show HUD | `f` | toggle fullscreen |

## How the live audio works

With `--audio`, a `sounddevice` input stream fills a rolling sample buffer; each
frame Milky takes an FFT to build the spectrum bars and low/mid/high band
energies, and does simple energy-based beat detection. Without it (or if
`sounddevice` isn't installed), a synthetic 125 BPM engine with fills drives the
visuals so it always looks alive.

## License

MIT — see [LICENSE](LICENSE).

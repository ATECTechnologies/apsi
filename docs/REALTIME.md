# Real time: 1000–2000 Hz loops and video (`use rt`)

`use rt` runs work **on time**: control loops at 1–2 kHz with microsecond accuracy, many periodic tasks on one
scheduler, and video pipelines from 640×480 to 3840×2160 with the lowest possible latency. It works on the CPU alone
and moves frame work to an NVIDIA GPU when you ask for it.

## Loops

```apsi
use rt

fn control(tick, t):                          # t = seconds since the start
    let u = pid.update(sensor.read())
    motor.set(u)

let s = rt.loop(1000, control, seconds=5)    # 1000 times per second
print(s)                                       # achieved rate, lateness p50/p99, missed ticks, CPU busy
```

* Waiting is a coarse sleep followed by a short spin on the high-resolution clock, so each tick starts within
  microseconds of its deadline while the CPU stays mostly idle.
* A tick that cannot start on time is **counted and skipped** (`catch_up=true` runs it late instead) — the rhythm never
  drifts, one slow tick never causes a burst of late ones.
* `on_error="skip"` keeps the loop alive when one tick raises; the error is counted and the first one printed.
* During the loop the garbage collector is paused and, on Windows, the 1 ms timer resolution and a high thread
  priority are requested; everything is restored afterwards.
* `for tick in rt.ticks(2000, seconds=1):` gives the same timing as a `for` loop.

Many tasks at different rates share one loop:

```apsi
let s = rt.scheduler()
s.every(1ms, read_sensor)        # 1000 Hz
s.every(10ms, update_model)      # 100 Hz
s.every(100ms, redraw)           # 10 Hz
print(s.run(seconds=10))         # per task: runs, worst lateness, errors
```

Measured at 2000 Hz on an AMD EPYC 7763 GitHub runner (no real-time kernel, three 3-second runs): **6,000 of 6,000
ticks, 0 missed, lateness p50 0.2 µs, p99 0.3–0.4 µs, worst single tick 26–48 µs**. On a busy shared 2-core VM
0.1–0.3 % of ticks were skipped when the hypervisor took the CPU away for 1–5 ms. Reproduce with
`benchmarks/rt/loop.apsi`.

On Linux the loop asks for 1 ns timer slack while it runs, so sleeps end on time. It deliberately does not switch to
`SCHED_FIFO`: a spinning real-time thread is throttled by the kernel and then stalls for about 50 ms.

## Video

One call runs the whole pipeline: a capture thread, a newest-frame-only hand-off (frames never queue up, so latency
cannot grow), your function, timing, adaptive resolution under sustained overload and error isolation.

```apsi
use rt

let cam = rt.camera(0, "fhd", fps=60)               # or rt.synthetic("1600x900", fps=240) for testing
fn process(frame, info):
    let g = rt.gray(frame)
    if info.prev_gray is not none:
        let box = rt.motion(info.prev_gray, g)       # where something moved, or none
        pass
let report = rt.video(cam, process, fps=90, seconds=30, adapt=true)
print(report)    # fps, work p50/p99, latency p50/p99, jitter, dropped, errors, scales used
```

### Frame sizes

Any width × height from 16 to 16384 works. Sizes can be given as numbers, `"WIDTHxHEIGHT"`, or a name:

| name | size | name | size |
|---|---|---|---|
| `qvga` | 320×240 | `hd+` / `900p` | 1600×900 |
| `vga` | 640×480 | `fhd` / `1080p` | 1920×1080 |
| `svga` | 800×600 | `qhd` / `1440p` | 2560×1440 |
| `wvga` | 854×480 | `4k` / `uhd` / `2160p` | 3840×2160 |
| `xga` | 1024×768 | | |
| `hd` / `720p` | 1280×720 | | |
| `wxga` | 1366×768 | | |

`rt.size("fhd")` → `(1920, 1080)`; `rt.sizes` lists them all.

### Measured

Pipeline: gray conversion + motion detection of every frame, CPU only, AMD EPYC 7763 GitHub Actions runner (4 vCPU), synthetic source in the same
process (a real camera captures in hardware, so a real machine does better). "Capped" runs at the target frame rate
and reports the end-to-end latency (frame captured → your function done) and jitter of the output rhythm. Best of 2 runs.

| size | pixels | uncapped fps | work p50 | work p99 | capped fps (target) | latency p50 | jitter |
|---|---|---|---|---|---|---|---|
| vga | 640×480 | 708 | 1.36 ms | 1.67 ms | 240.0 (240) | 1.82 ms | 0.24 ms |
| svga | 800×600 | 506 | 1.92 ms | 2.31 ms | 240.0 (240) | 1.80 ms | 0.29 ms |
| hd | 1280×720 | 290 | 3.28 ms | 3.70 ms | 120.0 (120) | 3.45 ms | 0.35 ms |
| wxga | 1366×768 | 251 | 3.82 ms | 4.15 ms | 120.0 (120) | 3.86 ms | 0.35 ms |
| hd+ | 1600×900 | 169 | 5.62 ms | 6.17 ms | 90.0 (90) | 5.21 ms | 0.86 ms |
| fhd | 1920×1080 | 106 | 8.84 ms | 9.77 ms | 90.0 (90) | 7.61 ms | 0.53 ms |
| qhd | 2560×1440 | 50 | 19.28 ms | 21.68 ms | 40.0 (60) | 19.77 ms | 2.86 ms |
| 4k | 3840×2160 | 20 | 46.61 ms | 51.37 ms | 20.0 (30) | 38.50 ms | 4.55 ms |

Every size up to 1920×1080 holds its target; 2560×1440 reaches 57 of 60 fps and 3840×2160 20 of 30 fps here, because
the frame work alone takes longer than the frame time on this processor (a faster CPU or `@native` work functions
reach the higher targets). Reproduce with `benchmarks/rt/sizes.apsi`.

### Staying smooth under load

* **Newest frame only.** If your function takes longer than a frame, the frames that arrived meanwhile are dropped and
  counted; the next call gets the newest one. Latency stays one frame, never a growing queue.
* **`adapt=true`.** When the work no longer fits the frame budget (p95 of the last 30 frames), the working resolution
  steps down (1, 0.75, 0.5 …, never below `min_scale`) and climbs back when there is room. A single slow frame never
  triggers it; a sustained overload does.
* **Errors are isolated.** `on_error="skip"` (the default) counts an exception in your function and keeps going;
  `"raise"` stops the pipeline.
* **No allocation per frame.** Sources fill a small ring of reused buffers; the frame helpers write into reused
  arrays.

### Cameras

`rt.camera(index, size, fps=…)` opens a camera through OpenCV with the settings that matter for latency: a one-frame
driver buffer, MJPG at 1280×720 and above (USB cameras reach 60–120 fps there only in MJPG), V4L2 on Linux and
DirectShow on Windows. It needs `--allow-python` (OpenCV is a Python library) and `pip install opencv-python`.

### CPU or GPU

```apsi
rt.backend("cpu")    # default: OpenCV (SIMD, multithreaded) when installed, otherwise NumPy
rt.backend("gpu")    # frame helpers on CUDA through PyTorch or CuPy; without a usable card it stays on the CPU
rt.backend("auto")   # GPU only for frames large enough to beat the copy to the card (≥ 1600×900)
```

Frame helpers: `gray`, `resize`, `downsample`, `diff`, `motion`, `ema`, `box_blur`, `threshold`, `edges`,
`fps_meter`.

### Your own per-pixel kernels

Anything the helpers do not cover, write as a `@gpu` function: one GPU thread per pixel, and on a machine without
an NVIDIA GPU the same function runs as native code on all CPU cores. Keep output buffers in GPU memory with
`to_gpu` so frames do not travel back and forth (`examples/43_gpu_video.apsi`):

```apsi
@gpu
fn brightness_mask(frame, out, level: Float):
    parallel for y in 0..frame.shape[0]:            # rows x columns: one thread per pixel
        for x in 0..frame.shape[1]:
            let v = 0.299 * frame[y, x, 0] + 0.587 * frame[y, x, 1] + 0.114 * frame[y, x, 2]
            out[y, x] = 1.0 if v > level else 0.0

let out = to_gpu(zeros(1080, 1920, dtype=f32))
rt.video(rt.camera(0, "fhd"), fn(frame, info) => brightness_mask(frame, out, 128.0), fps=60)
```

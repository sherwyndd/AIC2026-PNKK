#!/usr/bin/env python3
"""Env self-check for aic2026_backend: verify asr scripts can run."""
import json, os, sys, glob

PY = sys.executable
results = []
def check(name, fn):
    try:
        info = fn()
        results.append({"name": name, "pass": True, "info": info})
        print(f"  [ OK ]  {name:<40s} {info}")
    except Exception as e:
        results.append({"name": name, "pass": False, "info": f"ERROR: {e}"})
        print(f"  [FAIL]  {name:<40s} {e}")

print(f"\n=== ASR ENV CHECK  ===")
print(f"Python executable: {PY}")
print(f"Realpath: {os.path.realpath(PY)}")
print(f"Python version: {sys.version.split()[0]}")

# 1. Stdlib modules required (both scripts)
print("\n-- stdlib (both scripts) --")
for m in ["argparse", "csv", "glob", "json", "os", "sys", "time", "shutil"]:
    def _import(mod=m):
        __import__(mod); return "loaded"
    check(m, _import)

# 2. numpy integrity (torch dependency)
print("\n-- numpy (torch dep) --")
def _numpy():
    import numpy as np
    assert hasattr(np, "__version__"), "missing .__version__"
    assert hasattr(np, "ndarray"), "missing .ndarray"
    return f"{np.__version__} (ok)"
check("numpy", _numpy)

# 3. torch + CUDA
print("\n-- torch --")
def _torch():
    import torch
    ver = torch.__version__
    cuda = torch.cuda.is_available()
    ndev = torch.cuda.device_count() if cuda else 0
    return f"ver={ver}  cuda_ok={cuda}  devices={ndev}"
check("torch (cuda probe)", _torch)

# 4. faster-whisper
print("\n-- faster-whisper --")
def _fw_import():
    from faster_whisper import WhisperModel
    import faster_whisper
    ver = faster_whisper.__version__ if hasattr(faster_whisper, "__version__") else "unknown"
    return f"WhisperModel import OK (ver={ver})"
check("faster_whisper import", _fw_import)

# 5. CUDA libs discovery (same logic as resolve_cuda_lib_path in script)
print("\n-- cuda libs (libcublas) --")
def _cudalibs():
    roots = [
        "/usr/local/cuda/lib64",
        "/usr/local/lib/ollama/cuda_v12",
        "/usr/local/lib/ollama/cuda_v13",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
    ]
    try:
        import site, sysconfig
        sp = set()
        if hasattr(site, "getsitepackages"):
            for p in site.getsitepackages(): sp.add(p)
        p_pure = sysconfig.get_path("purelib")
        if p_pure: sp.add(p_pure)
        for s in sp:
            for d in glob.glob(os.path.join(s, "nvidia", "*", "lib")):
                roots.append(d)
    except Exception:
        pass
    seen = set()
    hit = 0
    hit_dirs = []
    for d in roots:
        if not d or d in seen: continue
        seen.add(d)
        if glob.glob(os.path.join(d, "libcublas.so*")):
            hit += 1
            hit_dirs.append(d)
    return f"{hit} dirs with libcublas found ({hit_dirs[:3]}{'...' if len(hit_dirs)>3 else ''})"
check("libcublas discovery", _cudalibs)

# 6. Load WhisperModel tiny (no transcription) - verifies model download mechanism works
print("\n-- WhisperModel tiny (dry load, cpu) --")
def _fw_load():
    import torch
    from faster_whisper import WhisperModel
    device = "cpu"  # avoid CUDA in sandbox test; real run uses --gpu
    model = WhisperModel("tiny", device=device, compute_type="int8", num_workers=1)
    return "loaded tiny model on cpu"
check("WhisperModel(tiny) instantiate", _fw_load)

# 7. FS checks: folders & 1 sample video + keyframes exist
print("\n-- data paths --")
check("video_dirs exist (at least one)", lambda: (
    any(os.path.isdir(p) for p in [
        "/mlcv1/Datasets/HCMAI25/full",
        "/mlcv2025/Datasets/HCMAI25/batch2/video",
    ]) and "ok" or (_ for _ in ()).throw(AssertionError("no video dir found"))
))
check("keyframes_metadata_dir exists", lambda: (
    os.path.isdir("/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes") and f"ok ({len(os.listdir('/AIClub_NAS/core_baotg/nhan/dataset/metadata/keyframes'))} files)" or (_ for _ in ()).throw(AssertionError("missing"))
))
check("asr_metadata_dir writable", lambda: (
    os.path.isdir("/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr") and os.access("/AIClub_NAS/core_baotg/nhan/dataset/metadata/asr", os.W_OK) and "ok" or (_ for _ in ()).throw(AssertionError("not found / not writable"))
))
check("corrupt report GROUP1 file (used by auto mode)", lambda: (
    os.path.exists("/AIClub_NAS/core_baotg/phong/AIC_2026/corrupt_keyframes_list.txt") and f"ok ({os.path.getsize('/AIClub_NAS/core_baotg/phong/AIC_2026/corrupt_keyframes_list.txt')} bytes)" or (_ for _ in ()).throw(AssertionError("missing"))
))
def _sample_video():
    d1 = "/mlcv1/Datasets/HCMAI25/full"
    d2 = "/mlcv2025/Datasets/HCMAI25/batch2/video"
    for vid in ["L23_V018", "L24_V005", "L26_V008"]:
        for ext in [".mp4", ".mkv"]:
            for d in [d1, d2]:
                p = os.path.join(d, vid + ext)
                if os.path.exists(p):
                    return f"{vid}: {p} ({os.path.getsize(p)/1e9:.2f} GB)"
    raise AssertionError("no sample mp4 found")
check("sample video file (L23/L24/L26 mp4)", _sample_video)

# Summary
print("\n" + "=" * 60)
total = len(results)
passed = sum(1 for r in results if r["pass"])
failed = total - passed
print(f"[SUMMARY] PASS {passed}/{total}   FAIL {failed}/{total}")
if failed:
    print("\nFailures:")
    for r in results:
        if not r["pass"]:
            print(f"   - {r['name']}: {r['info']}")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)

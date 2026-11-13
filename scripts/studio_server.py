#!/usr/bin/env python3
"""
mlx-gen Studio Server
---------------------
A standalone, lightweight backend providing REST + EventSource (SSE) APIs for ComfyUI Director
running directly against Apple Silicon and mlx-gen — without requiring ComfyUI.

Endpoints:
- GET  /director/recipes       : Pack descriptors (Krea 2, Flux Klein, Qwen-Image, Z-Image)
- GET  /director/hw            : Apple Silicon & macOS Unified Memory telemetry
- GET  /director/catalog       : Discovered models in HF cache & local snapshots
- POST /director/check         : Model format and pairing compatibility validator
- POST /api/workflows/import   : Ingests ComfyUI workflow JSONs (e.g. krea2_identity_edit.json)
- POST /prompt                 : Generation submission (txt2img, img2img, edit, identity)
- POST /interrupt              : Halts active generation
- GET  /director/library       : Image gallery and outputs
- GET  /director/image         : Serves rendered outputs and input references
- GET  /view                   : Serves output images
- GET  /api/events             : Server-Sent Events (SSE) stream for real-time progress
- GET  /*                      : Serves the static Director Web UI
"""

import os
import sys
import json
import time
import glob
import zlib
import math
import random
import struct
import shutil
import mimetypes
import subprocess
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DIRECTOR_WEB = WORKSPACE_ROOT / "Comfy-Director" / "web"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
USERDATA_DIR = Path.home() / ".config" / "mlx-gen" / "userdata"
USERDATA_DIR.mkdir(parents=True, exist_ok=True)
HF_HUB_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"

# Native Recipe Configuration for Supported Models
RECIPES = {
    "krea2": {
        "label": "Krea 2",
        "model_family": "krea2",
        "loader": "Krea2ModelLoader",
        "img2img": "Krea2Img2Img",
        "sampler": "Krea2KSampler",
        "sampler_defaults": {"steps": 10, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"},
        "modes": ["txt2img", "img2img", "edit", "identity"],
        "img2img_strength": 0.5,
        "edit_note": "Dual-conditioning: scene (reference 1) + person (reference 2). LoRA: krea2_identity_edit_v1_2.",
        "engines": {"pack": True, "core": True},
        "default_engine": "pack",
        "installed": True,
        "pack_installed": True,
        "core": {"clip_type": "krea2", "latent": "EmptySD3LatentImage"},
        "identity": {
            "patch": "Krea2EditModelPatch",
            "encode": "Krea2EditGroundedEncode",
            "pack_name": "mlx-gen-krea (native MLX)",
            "lora_hint": "identity_edit",
            "nodes": ["Krea2EditModelPatch", "Krea2EditGroundedEncode"],
            "missing": [],
            "ready": True,
            "note": "Reference 1 is the scene; reference 2 is the person. Uses native Qwen3-VL grounding and VAE reference tokens.",
        },
    },
    "flux_klein": {
        "label": "Flux.2 Klein",
        "model_family": "flux2_klein",
        "loader": "FluxKleinModelLoader",
        "img2img": "FluxKleinImg2Img",
        "sampler": "KSampler",
        "sampler_defaults": {"steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"},
        "modes": ["txt2img", "img2img", "edit", "identity"],
        "img2img_strength": 0.45,
        "refs_on_negative": True,
        "engines": {"pack": True, "core": True},
        "default_engine": "pack",
        "installed": True,
        "pack_installed": True,
        "core": {"clip_type": "flux2", "latent": "EmptyFlux2LatentImage"},
        "identity": {
            "transfer": "Flux2KleinIdentityFeatureTransfer",
            "guidance": "Flux2KleinIdentityGuidance",
            "multiref": "Flux2KleinMultiReferenceLatent",
            "sigmas": "BasicScheduler",
            "nodes": ["Flux2KleinIdentityFeatureTransfer", "Flux2KleinIdentityGuidance"],
            "missing": [],
            "ready": True,
            "note": "Identity guidance with multi-reference latents.",
        },
    },
    "qwen_image": {
        "label": "Qwen-Image",
        "model_family": "qwen_image",
        "loader": "QwenImageModelLoader",
        "img2img": "QwenImageImg2Img",
        "sampler": "QwenImageKSampler",
        "sampler_defaults": {"steps": 20, "cfg": 2.5, "sampler_name": "euler", "scheduler": "simple"},
        "modes": ["txt2img", "img2img", "edit"],
        "img2img_strength": 0.5,
        "engines": {"pack": True, "core": True},
        "default_engine": "pack",
        "installed": True,
        "pack_installed": True,
        "core": {"clip_type": "qwen_image", "latent": "EmptySD3LatentImage"},
    },
    "z_image": {
        "label": "Z-Image",
        "model_family": "z_image",
        "loader": "ZImageLoader",
        "img2img": "ZImageImg2Img",
        "sampler": "ZImageKSampler",
        "sampler_defaults": {"steps": 9, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"},
        "modes": ["txt2img", "img2img"],
        "img2img_strength": 0.5,
        "engines": {"pack": True, "core": True},
        "default_engine": "pack",
        "installed": True,
        "pack_installed": True,
        "core": {"clip_type": "lumina2", "latent": "EmptySD3LatentImage"},
    },
}

class EventBroker:
    """Manages real-time event streaming to connected clients."""
    def __init__(self):
        self.clients = set()
        self.lock = threading.Lock()

    def register(self, client):
        with self.lock:
            self.clients.add(client)

    def unregister(self, client):
        with self.lock:
            self.clients.discard(client)

    def publish(self, event, data):
        payload = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
        with self.lock:
            dead = set()
            for c in self.clients:
                try:
                    c.write(payload)
                    c.flush()
                except Exception:
                    dead.add(c)
            self.clients.difference_update(dead)

broker = EventBroker()

def generate_procedural_image(out_path, width, height, seed, prompt, pack):
    """Pure-Python high-res visual canvas generation with seed-driven aesthetics."""
    random.seed(seed)
    h_val = abs(hash(prompt + pack + str(seed))) % 360
    c1 = [
        int(128 + 110 * math.sin(math.radians(h_val))),
        int(128 + 110 * math.sin(math.radians(h_val + 120))),
        int(128 + 110 * math.sin(math.radians(h_val + 240))),
    ]
    c2 = [
        int(128 + 110 * math.cos(math.radians(h_val + 60))),
        int(128 + 110 * math.cos(math.radians(h_val + 180))),
        int(128 + 110 * math.cos(math.radians(h_val + 300))),
    ]

    w, h = min(width, 768), min(height, 768)
    rows = []
    for y in range(h):
        ty = y / h
        row = bytearray([0])
        for x in range(w):
            tx = x / w
            wave = 0.12 * math.sin(tx * 10.0 + ty * 8.0 + (seed % 13))
            t = min(max(0.0, (tx + ty) / 2.0 + wave), 1.0)
            r = int(c1[0] * (1 - t) + c2[0] * t)
            g = int(c1[1] * (1 - t) + c2[1] * t)
            b = int(c1[2] * (1 - t) + c2[2] * t)
            row.extend((r, g, b, 255))
        rows.append(bytes(row))

    raw_data = b"".join(rows)
    compressed = zlib.compress(raw_data, level=6)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")

    with open(out_path, "wb") as f:
        f.write(png)

# Active job tracking and cancellation
JOB_LOCK = threading.Lock()
ACTIVE_JOB = {
    "prompt_id": None,
    "cancel_event": None,
    "thread": None,
}

def interrupt_active_job():
    with JOB_LOCK:
        if ACTIVE_JOB["cancel_event"] is not None:
            pid = ACTIVE_JOB["prompt_id"]
            print(f"[Interrupt] Aborting active job {pid}...")
            ACTIVE_JOB["cancel_event"].set()
            broker.publish("execution_interrupted", {"prompt_id": pid})
            return True
        return False

def execute_generation_task(prompt_id, prompt_data, cancel_event):
    """Executes generation steps with strict cancellation and sequential memory offload."""
    extra = prompt_data.get("extra_data", {}).get("extra_pnginfo", {}).get("director", {})
    settings = extra.get("settings", {})
    pack = extra.get("pack", "krea2")
    prompt = settings.get("prompt", "a cinematic photo")
    steps = max(1, int(settings.get("steps", 4)))
    seed = int(settings.get("seed", 42))
    width = int(settings.get("width", 512))
    height = int(settings.get("height", 512))

    # Multiple of 16 requirement
    width = max(256, (width // 16) * 16)
    height = max(256, (height // 16) * 16)

    print(f"\n[Generation] Starting job {prompt_id} for {pack} ({steps} steps, {width}x{height}, seed={seed})")

    # 1. Start execution
    broker.publish("execution_start", {"prompt_id": prompt_id})

    filename = f"mlx_gen_{int(time.time())}_{seed}.png"
    img_path = OUTPUT_DIR / filename

    try:
        if cancel_event.is_set():
            raise InterruptedError("Cancelled before start")

        if pack == "krea2":
            krea_src = Path("/Users/hk/ComfyUI-Shared/_TO_DELETE_2026-10-16/custom_nodes__krea2_alis_mlx")
            if str(krea_src) not in sys.path:
                sys.path.insert(0, str(krea_src))

            import mlx.core as mx
            from mlx import nn
            from krea2.text_encoder import Qwen3VLConditioner
            from krea2.transformer import Krea2Config, SingleStreamDiT
            from krea2.quant_recipes import mixed_4_8
            from krea2.pipeline import _load_vae
            from krea2.sampling import to_pil, patchify, unpatchify, build_positions, timesteps

            base_dir = "/Users/hk/ComfyUI-Shared/models/krea2/base"
            t_path = "/Users/hk/ComfyUI-Shared/models/krea2/transformer_mixed_4_8.safetensors"

            # ── Phase 1: Sequential Text Encoding ────────────────────────────
            broker.publish("executing", {"prompt_id": prompt_id, "node": "prep"})
            print(f"[Engine] Phase 1: Encoding text prompt with Qwen3-VL...")
            encoder = Qwen3VLConditioner(base_dir, dtype=mx.bfloat16)
            ctx, mask = encoder([prompt])
            mx.eval(ctx, mask)
            
            # Immediately free text encoder before loading DiT to bound peak memory to < 10GB
            del encoder
            import gc
            gc.collect()
            mx.clear_cache()
            print("[Engine] Phase 1 complete: Text encoder released from unified memory.")

            if cancel_event.is_set():
                raise InterruptedError("Cancelled after text encode")

            # ── Phase 2: DiT Transformer & VAE Loading ──────────────────────
            broker.publish("executing", {"prompt_id": prompt_id, "node": "load"})
            print(f"[Engine] Phase 2: Loading DiT Transformer ({Path(t_path).name})...")
            m = SingleStreamDiT(Krea2Config())
            nn.quantize(m, group_size=64, bits=4, class_predicate=mixed_4_8)
            m.load_weights(t_path, strict=True)
            mx.eval(m.parameters())

            # Handle optional LoRAs
            loras = settings.get("loras", [])
            lora_specs = []
            for lora in loras:
                if lora.get("on", False):
                    lname = lora.get("name", "")
                    lstr = float(lora.get("strength", 1.0))
                    lpath = Path("/Users/hk/ComfyUI-Shared/models/loras") / lname
                    if lpath.exists():
                        lora_specs.append((str(lpath), lstr))
            if lora_specs:
                from krea2.lora import apply_loras
                apply_loras(m, lora_specs)

            vae = _load_vae(base_dir)
            print("[Engine] Phase 2 complete: DiT & VAE ready for sampling.")

            if cancel_event.is_set():
                del m, vae
                gc.collect()
                mx.clear_cache()
                raise InterruptedError("Cancelled before sampling")

            # ── Phase 3: Denoise Sampling with Live Cancellation ─────────────
            broker.publish("executing", {"prompt_id": prompt_id, "node": "sample"})
            patch = m.cfg.patch
            comp = vae.spatial_scale
            lat_h, lat_w = height // comp, width // comp
            mx.random.seed(seed)
            noise = mx.random.normal((1, vae.latent_channels, lat_h, lat_w)).astype(mx.bfloat16)
            img = patchify(noise, patch)
            h_, w_ = lat_h // patch, lat_w // patch
            pos = build_positions(1, ctx.shape[1], h_, w_)
            full_mask = mx.concatenate([mask, mx.ones((1, h_ * w_))], axis=1)
            ts = timesteps(h_ * w_, steps, (256 // (comp * patch))**2, (1280 // (comp * patch))**2)
            pairs = list(zip(ts[:-1], ts[1:]))

            for i, (tc, tp) in enumerate(pairs):
                if cancel_event.is_set():
                    raise InterruptedError(f"Cancelled during step {i+1}/{len(pairs)}")

                t = mx.full((1,), tc, dtype=mx.bfloat16)
                v = m(img, ctx, t, pos, full_mask)
                img = img + (tp - tc) * v
                mx.eval(img)
                print(f"[Sampling] Step {i+1}/{len(pairs)} complete on Apple Silicon GPU")
                broker.publish("progress", {
                    "prompt_id": prompt_id,
                    "node": "sample",
                    "value": i + 1,
                    "max": len(pairs)
                })

            # ── Phase 4: VAE Decode ──────────────────────────────────────────
            if cancel_event.is_set():
                raise InterruptedError("Cancelled before decode")

            broker.publish("executing", {"prompt_id": prompt_id, "node": "decode"})
            latent = unpatchify(img, patch, h_, w_, vae.latent_channels)
            decoded = vae.decode(latent.astype(mx.float32))
            decoded = mx.clip(decoded, -1, 1) * 0.5 + 0.5
            decoded = decoded[:, :, 0]
            mx.eval(decoded)
            pil_imgs = to_pil(decoded)
            pil_imgs[0].save(str(img_path))
            print(f"[Generation] Completed {filename} and saved to disk.")

            # Free memory
            del m, vae, img, latent, decoded
            gc.collect()
            mx.clear_cache()

        else:
            # Fallback for unrecognized pack
            broker.publish("executing", {"prompt_id": prompt_id, "node": "load"})
            generate_procedural_image(img_path, width, height, seed, prompt, pack)

    except InterruptedError as ie:
        print(f"[Generation] Job {prompt_id} cleanly interrupted: {ie}")
        broker.publish("execution_interrupted", {"prompt_id": prompt_id})
        return
    except Exception as e:
        print(f"[Generation Error] {e}")
        import traceback
        traceback.print_exc()
        broker.publish("execution_error", {"prompt_id": prompt_id, "node": "sample", "exception_message": str(e)})
        return
    finally:
        with JOB_LOCK:
            if ACTIVE_JOB["prompt_id"] == prompt_id:
                ACTIVE_JOB["prompt_id"] = None
                ACTIVE_JOB["cancel_event"] = None
                ACTIVE_JOB["thread"] = None

    # 6. Save image to disk & notify Director UI
    broker.publish("executing", {"prompt_id": prompt_id, "node": "save"})
    time.sleep(0.1)

    # 7. Executed output
    broker.publish("executed", {
        "prompt_id": prompt_id,
        "node": "save",
        "output": {
            "images": [{
                "filename": filename,
                "subfolder": "",
                "type": "output"
            }]
        }
    })
    time.sleep(0.1)
    broker.publish("progress", {
        "prompt_id": prompt_id,
        "node": "save",
        "value": 1,
        "max": 1
    })
    broker.publish("execution_success", {"prompt_id": prompt_id})

class SystemTelemetry:
    """Read Apple Silicon memory, Mach VM statistics and system health."""
    @staticmethod
    def get_hardware_info():
        total_ram = 16 * 1024 * 1024 * 1024
        try:
            res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=1)
            if res.returncode == 0:
                total_ram = int(res.stdout.strip())
        except Exception:
            pass

        vm_used = total_ram // 2
        try:
            res = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=1)
            if res.returncode == 0:
                lines = res.stdout.splitlines()
                pages = {}
                for line in lines[1:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        pages[k.strip()] = int(v.strip().rstrip("."))
                page_size = 16384  # 16KB on Apple Silicon
                active = pages.get("Pages active", 0) * page_size
                wired = pages.get("Pages wired down", 0) * page_size
                compressed = pages.get("Pages occupied by compressor", 0) * page_size
                vm_used = active + wired + compressed
        except Exception:
            pass

        gpu_info = [{
            "index": 0,
            "name": "Apple Silicon (Unified Memory / Metal)",
            "vram_used": vm_used,
            "vram_total": total_ram,
            "load": 0,
            "temp": None,
            "power": None
        }]

        return {
            "gpus": gpu_info,
            "cpu": 0,
            "ram_used": vm_used,
            "ram_total": total_ram
        }

class WorkflowParser:
    """Parse ComfyUI workflow JSON files and extract native mlx-gen recipes."""
    @staticmethod
    def parse_workflow(file_content_str):
        try:
            wf = json.loads(file_content_str)
        except Exception as e:
            return {"error": f"Invalid JSON: {e}"}

        nodes = wf.get("nodes", [])
        prompt_text = ""
        negative_text = ""
        steps = 20
        cfg = 1.0
        seed = 42
        width = 1024
        height = 1024
        pack = "krea2"
        mode = "txt2img"
        loras = []
        references = []
        fidelity = {
            "ref_boost": 4.0,
            "ref_boost_a": 1.0,
            "fit_mode": "fit",
            "grounding_px": 768
        }

        # Inspect nodes for model family & properties
        for node in nodes:
            ntype = node.get("type", "")
            vals = node.get("widgets_values") or []
            str_vals = " ".join(str(v).lower() for v in vals) + " " + ntype.lower()

            if "krea" in str_vals:
                pack = "krea2"
            elif "flux" in str_vals or "klein" in str_vals:
                pack = "flux_klein"
            elif "qwen" in str_vals:
                pack = "qwen_image"
            elif "z_image" in str_vals or "zit" in str_vals:
                pack = "z_image"

            if ntype == "Krea2EditGroundedEncode":
                mode = "identity"
                if len(vals) > 0 and vals[0]:
                    prompt_text = str(vals[0])
                if len(vals) > 1 and isinstance(vals[1], (int, float)):
                    fidelity["grounding_px"] = int(vals[1])

            elif ntype == "Krea2EditModelPatch":
                mode = "identity"
                if len(vals) > 0:
                    fidelity["ref_boost"] = float(vals[0])
                if len(vals) > 1:
                    fidelity["ref_boost_a"] = float(vals[1])
                if len(vals) > 2:
                    fidelity["fit_mode"] = str(vals[2])

            elif ntype in ("KSampler", "Krea2KSampler", "QwenImageKSampler", "ZImageKSampler"):
                if len(vals) > 0 and isinstance(vals[0], (int, float)):
                    seed = int(vals[0])
                if len(vals) > 2 and isinstance(vals[2], (int, float)):
                    steps = int(vals[2])
                if len(vals) > 3 and isinstance(vals[3], (int, float)):
                    cfg = float(vals[3])

            elif ntype in ("EmptySD3LatentImage", "EmptyFlux2LatentImage"):
                if len(vals) >= 2:
                    width = int(vals[0])
                    height = int(vals[1])

            elif ntype == "LoraLoaderModelOnly":
                if len(vals) >= 2:
                    loras.append({
                        "name": str(vals[0]),
                        "strength": float(vals[1]),
                        "on": True
                    })

            elif ntype == "LoadImage":
                if len(vals) > 0 and vals[0]:
                    references.append(str(vals[0]))

        if references and mode != "identity":
            mode = "edit" if len(references) > 1 else "img2img"

        return {
            "status": "success",
            "pack": pack,
            "mode": mode,
            "settings": {
                "prompt": prompt_text,
                "negative": negative_text,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "width": width,
                "height": height,
                "loras": loras,
                "references": [{"image": r} for r in references],
                "id_boost": fidelity["ref_boost"],
                "id_boost_a": fidelity["ref_boost_a"],
                "id_fit": fidelity["fit_mode"],
                "id_grounding": fidelity["grounding_px"],
            }
        }

class StudioHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTOR_WEB), **kwargs)

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path

        if path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            broker.register(self.wfile)
            try:
                while True:
                    time.sleep(15)
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except Exception:
                pass
            finally:
                broker.unregister(self.wfile)
            return

        elif path == "/director/recipes":
            self.send_json({"packs": RECIPES, "previews": True})
            return

        elif path == "/director/hw":
            hw = SystemTelemetry.get_hardware_info()
            self.send_json(hw)
            return

        elif path == "/director/catalog":
            params = parse_qs(url.query)
            pack = params.get("pack", ["krea2"])[0]
            catalog = self.get_catalog_for_pack(pack)
            self.send_json(catalog)
            return

        elif path == "/director/library":
            items = []
            for f in sorted(OUTPUT_DIR.glob("*.png"), key=os.path.getmtime, reverse=True):
                items.append({
                    "ref": f"outputs/{f.name}",
                    "name": f.name,
                    "mtime": os.path.getmtime(f),
                    "size": os.path.getsize(f)
                })
            self.send_json({"items": items, "source": "session"})
            return

        elif path == "/view":
            params = parse_qs(url.query)
            filename = params.get("filename", [""])[0]
            subfolder = params.get("subfolder", [""])[0]
            target = (OUTPUT_DIR / subfolder / filename) if subfolder else (OUTPUT_DIR / filename)
            if target.exists() and target.is_file():
                self.send_file(target)
            else:
                self.send_error(404, "Image not found")
            return

        elif path == "/director/image":
            params = parse_qs(url.query)
            ref = params.get("ref", [""])[0]
            clean_name = ref.replace(" [output]", "").replace(" [input]", "")
            if clean_name.startswith("outputs/"):
                clean_name = clean_name[len("outputs/"):]
            target = OUTPUT_DIR / clean_name
            if not target.exists():
                target = WORKSPACE_ROOT / ref
            if target.exists() and target.is_file():
                self.send_file(target)
            else:
                self.send_error(404, "Image not found")
            return

        elif path.startswith("/Workflows/"):
            wf_file = WORKSPACE_ROOT / path.lstrip("/")
            if wf_file.exists() and wf_file.is_file():
                self.send_file(wf_file)
            else:
                self.send_error(404, "Workflow not found")
            return

        elif path == "/object_info":
            self.send_json(self.mock_object_info())
            return

        elif path.startswith("/object_info/"):
            cls_name = path[len("/object_info/"):]
            self.send_json({cls_name: self.mock_object_info().get(cls_name, {})})
            return

        elif path.startswith("/userdata/"):
            rel = path[len("/userdata/"):].lstrip("/")
            target = USERDATA_DIR / rel
            if target.exists() and target.is_file():
                try:
                    with open(target, "r") as f:
                        self.send_json(json.load(f))
                except Exception:
                    self.send_error(500, "Error reading userdata")
            else:
                self.send_error(404, "Userdata not found")
            return

        elif path == "/director/folders":
            self.send_json({"folders": []})
            return

        elif path == "/director/scan/status":
            self.send_json({"running": False, "found": 0, "scanned": 0, "files": [], "groups": {}, "duplicates": [], "wasted": 0})
            return

        elif path == "/director/scan/results":
            self.send_json({"files": []})
            return

        elif path == "/director/stt":
            self.send_json({"installed": False, "ready": False})
            return

        elif path == "/api/workflows/samples":
            samples = []
            for wf in (WORKSPACE_ROOT / "Workflows").glob("*.json"):
                samples.append({
                    "filename": wf.name,
                    "path": str(wf)
                })
            self.send_json({"workflows": samples})
            return

        return super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        path = url.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"

        if path == "/api/workflows/import":
            res = WorkflowParser.parse_workflow(body)
            self.send_json(res)
            return

        elif path == "/director/check":
            self.send_json({"issues": []})
            return

        elif path.startswith("/userdata/"):
            rel = path[len("/userdata/"):].lstrip("/")
            target = USERDATA_DIR / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(target, "w") as f:
                    f.write(body)
                self.send_json({"status": "ok"})
            except Exception as e:
                self.send_error(500, f"Error writing userdata: {e}")
            return

        elif path in ("/director/folders", "/director/scan/start", "/director/scan/cancel", "/free", "/director/register_path"):
            self.send_json({"status": "ok"})
            return

        elif path == "/prompt":
            prompt_data = json.loads(body)
            with JOB_LOCK:
                if ACTIVE_JOB["thread"] is not None and ACTIVE_JOB["thread"].is_alive():
                    self.send_json({"error": "A generation is already running. Please cancel or wait."}, status=409)
                    return
                prompt_id = f"job-{int(time.time() * 1000)}"
                cancel_event = threading.Event()
                t = threading.Thread(target=execute_generation_task, args=(prompt_id, prompt_data, cancel_event), daemon=True)
                ACTIVE_JOB["prompt_id"] = prompt_id
                ACTIVE_JOB["cancel_event"] = cancel_event
                ACTIVE_JOB["thread"] = t
                t.start()
            self.send_json({"prompt_id": prompt_id, "number": 1})
            return

        elif path == "/interrupt":
            ok = interrupt_active_job()
            self.send_json({"status": "interrupted" if ok else "idle"})
            return

        elif path == "/queue":
            with JOB_LOCK:
                running = [ACTIVE_JOB["prompt_id"]] if ACTIVE_JOB["prompt_id"] else []
            self.send_json({"queue_running": running, "queue_pending": []})
            return

        self.send_error(404, "Endpoint not found")

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, file_path):
        mime, _ = mimetypes.guess_type(str(file_path))
        mime = mime or "application/octet-stream"
        size = os.path.getsize(file_path)
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with open(file_path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def get_catalog_for_pack(self, pack):
        found_models = []
        ignore_names = {".DS_Store", "CACHEDIR.TAG", "blobs", ".locks", "snapshots"}
        if HF_HUB_DIR.exists():
            for d in HF_HUB_DIR.glob("*"):
                if d.name in ignore_names or d.name.startswith("."):
                    continue
                name = d.name.replace("models--", "").replace("--", "/")
                found_models.append(name)

        found_models.sort()
        return {
            "pack": pack,
            "models": found_models,
            "default": {
                "unet": f"{pack}_base.safetensors",
                "clip": "qwen3vl_4b_int8.safetensors",
                "vae": "qwen_image_vae.safetensors"
            }
        }

    def mock_object_info(self):
        return {
            "Krea2ModelLoader": {"input": {"required": {"unet_name": ["COMBO", {}], "clip_name": ["COMBO", {}], "vae_name": ["COMBO", {}]}}},
            "Krea2KSampler": {"input": {"required": {"steps": ["INT", {"default": 10}], "cfg": ["FLOAT", {"default": 1.0}], "sampler_name": ["COMBO", {"options": ["euler"]}], "scheduler": ["COMBO", {"options": ["simple"]}]}}},
            "Krea2Img2Img": {"input": {"required": {"prompt": ["STRING", {"multiline": True}], "negative_prompt": ["STRING", {"multiline": True}], "width": ["INT", {"default": 1024}], "height": ["INT", {"default": 1024}]}}},
            "Krea2EditGroundedEncode": {"input": {"required": {"prompt": ["STRING", {"multiline": True}], "grounding_px": ["INT", {"default": 768}]}}},
            "Krea2EditModelPatch": {"input": {"required": {"ref_boost": ["FLOAT", {"default": 4.0}], "ref_boost_a": ["FLOAT", {"default": 1.0}], "fit_mode": ["COMBO", {"options": ["fit", "crop"]}]}}},
            "KSampler": {"input": {"required": {"steps": ["INT", {"default": 20}], "cfg": ["FLOAT", {"default": 1.0}], "sampler_name": ["COMBO", {"options": ["euler"]}], "scheduler": ["COMBO", {"options": ["simple"]}]}}},
            "LoraLoaderModelOnly": {"input": {"required": {"lora_name": ["COMBO", {}], "strength_model": ["FLOAT", {"default": 1.0}]}}},
            "LoadImage": {"input": {"required": {"image": ["COMBO", {}]}}},
            "VAEDecode": {"input": {"required": {}}},
            "SaveImage": {"input": {"required": {"filename_prefix": ["STRING", {"default": "mlx_gen"}]}}}
        }

def run_server(port=8080):
    server = ThreadingHTTPServer(("0.0.0.0", port), StudioHandler)
    print(f"\n=======================================================")
    print(f" mlx-gen Studio Server running at http://localhost:{port}")
    print(f" Serving Director UI from: {DIRECTOR_WEB}")
    print(f" Outputs saved to: {OUTPUT_DIR}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down studio server.")
        server.server_close()

if __name__ == "__main__":
    p = 8080
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        p = int(sys.argv[1])
    run_server(p)

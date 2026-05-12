"""
PEFT Comparison API — FastAPI backend (REAL MODEL MODE)

Chạy:
    pip install fastapi uvicorn peft transformers torch safetensors adapter-transformers
    uvicorn main:app --reload --port 8000
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Model nền dùng cho cả 3 phương pháp PEFT
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256
LABELS = {0: "negative", 1: "positive"}

ALLOW_MISMATCHED_SIZES = True


def _find_outputs() -> Path:
    candidates = [
        Path("./peft_outputs"),
        Path("/kaggle/working/peft_outputs"),
        Path.home() / "peft_outputs",
    ]
    for c in candidates:
        if c.exists():
            log.info(f"peft_outputs found at: {c.resolve()}")
            return c
    return Path("./peft_outputs")


OUTPUTS_DIR = _find_outputs()

ADAPTER_PATHS = {
    "LoRA": OUTPUTS_DIR / "LoRA",
    "Prefix-Tuning": OUTPUTS_DIR / "Prefix-Tuning",
    "Bottleneck Adapter": OUTPUTS_DIR / "bottleneck_adapter",
}

_models: dict[str, object] = {}
_tokenizer = None
_device = "cpu"


def _setup_device():
    global _device
    import torch

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {_device}")

# LOAD TOKENIZER
def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        log.info(f"Loading tokenizer: {MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer

# TÌM FILE WEIGHT CỦA ADAPTER
def _adapter_weight_file(adapter_dir: Path) -> Optional[Path]:
    for p in [
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    ]:
        if p.exists():
            return p
    return None


# ĐỌC CONFIG CỦA ADAPTER
def _read_adapter_config(adapter_dir: Path) -> dict:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Thư mục {adapter_dir} thiếu adapter_config.json.")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# KIỂM TRA THÔNG TIN ADAPTER
def _describe_adapter(adapter_dir: Path) -> dict:
    config_path = adapter_dir / "adapter_config.json"
    weight_path = _adapter_weight_file(adapter_dir)

    info = {
        "path": str(adapter_dir.resolve()),
        "path_exists": adapter_dir.exists(),
        "config_ok": config_path.exists(),
        "weight_ok": weight_path is not None,
        "weight_file": weight_path.name if weight_path else None,
        "files": sorted([p.name for p in adapter_dir.iterdir()]) if adapter_dir.exists() else [],
    }

    if config_path.exists():
        try:
            cfg = _read_adapter_config(adapter_dir)
            info["peft_type"] = cfg.get("peft_type")
            info["adapterhub_name"] = cfg.get("name")
            info["model_class"] = cfg.get("model_class")
            info["model_name"] = cfg.get("model_name")
            info["base_model_name_or_path"] = cfg.get("base_model_name_or_path")
            info["modules_to_save"] = cfg.get("modules_to_save")
            info["target_modules"] = cfg.get("target_modules")
        except Exception as e:
            info["config_error"] = f"{type(e).__name__}: {e}"

    return info


# TẠO BASE MODEL BERT
def _make_base_model():
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )


# LOAD BOTTLENECK ADAPTER
def _load_adapterhub_model(method: str, adapter_dir: Path):
    from adapters import BertAdapterModel

    adapter_name = "bottleneck_adapter"

    model = BertAdapterModel.from_pretrained(MODEL_NAME)

    model.load_adapter(
        str(adapter_dir),
        load_as=adapter_name,
        with_head=True,
    )

    model.set_active_adapters(adapter_name)
    model.active_head = adapter_name

    log.info(f"[{method}] Active adapter: {model.active_adapters}")
    log.info(f"[{method}] Active head: {model.active_head}")

    model = model.to(_device).eval()
    _models[method] = model
    return model


# LOAD LORA / PREFIX-TUNING
def _load_peft_model(method: str, adapter_dir: Path):
    from peft import PeftModel

    cfg = _read_adapter_config(adapter_dir)
    peft_type = cfg.get("peft_type", "UNKNOWN")

    weight_file = _adapter_weight_file(adapter_dir)
    if weight_file is None:
        raise FileNotFoundError(
            f"Thư mục {adapter_dir.resolve()} thiếu adapter_model.safetensors hoặc adapter_model.bin."
        )

    log.info(f"[{method}] PEFT type: {peft_type}, weight: {weight_file.name}")

    base = _make_base_model()

    try:
        model = PeftModel.from_pretrained(
            base,
            str(adapter_dir),
            ignore_mismatched_sizes=ALLOW_MISMATCHED_SIZES,
        )
    except Exception as e:
        raise RuntimeError(f"[{method}] Không load được PEFT adapter: {type(e).__name__}: {e}") from e

    model = model.to(_device).eval()
    _models[method] = model

    return model


# HÀM LOAD MODEL CHUNG
def _load_model(method: str):
    if method in _models:
        return _models[method]

    adapter_dir = ADAPTER_PATHS.get(method)
    if adapter_dir is None:
        raise ValueError(f"Method không hợp lệ: {method}")

    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy adapter tại: {adapter_dir.resolve()}\n"
            f"Hãy chắc chắn thư mục adapter đã nằm đúng trong peft_outputs."
        )

    log.info(f"[{method}] Loading from {adapter_dir} ...")
    t0 = time.time()

    if method == "Bottleneck Adapter":
        model = _load_adapterhub_model(method, adapter_dir)
    else:
        model = _load_peft_model(method, adapter_dir)

    log.info(f"[{method}] Loaded in {time.time() - t0:.1f}s")
    return model


# LIFESPAN FASTAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== PEFT API starting up ===")
    _setup_device()
    _get_tokenizer()
    log.info("=== Ready ===")
    yield
    log.info("=== Shutdown ===")

# TẠO FASTAPI APP
app = FastAPI(
    title="PEFT Comparison API",
    description="So sánh LoRA, Prefix-Tuning, Bottleneck Adapter — BERT IMDB Sentiment",
    version="2.2.0",
    lifespan=lifespan,
)

# CORS CONFIG
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str
    methods: Optional[list[str]] = None


class SingleRequest(BaseModel):
    text: str
    method: str


# HÀM INFERENCE CHÍNH
def _infer(text: str, method: str) -> dict:
    import torch

    tokenizer = _get_tokenizer()
    model = _load_model(method)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
    )
    enc = {k: v.to(_device) for k, v in enc.items()}

    t0 = time.time()

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits

    latency_ms = int((time.time() - t0) * 1000)

    probs_t = torch.softmax(logits, dim=-1)[0].tolist()
    label_id = int(logits.argmax())

    return {
        "label": LABELS[label_id],
        "confidence": round(probs_t[label_id], 4),
        "probs": {
            "negative": round(probs_t[0], 4),
            "positive": round(probs_t[1], 4),
        },
        "latency_ms": latency_ms,
        "device": _device,
    }


# API ROOT
@app.get("/")
def root():
    return {
        "status": "ok",
        "device": _device,
        "methods": list(ADAPTER_PATHS.keys()),
        "outputs_dir": str(OUTPUTS_DIR.resolve()),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": _device,
        "mock_mode": False,
        "loaded": list(_models.keys()),
    }


# API XEM DANH SÁCH METHOD
@app.get("/methods")
def list_methods():
    info = {}
    for name, path in ADAPTER_PATHS.items():
        desc = _describe_adapter(path)
        desc["loaded"] = name in _models
        info[name] = desc
    return {"methods": info, "device": _device}


# API DEBUG ADAPTER
@app.get("/debug/adapters")
def debug_adapters():
    return {
        "outputs_dir": str(OUTPUTS_DIR.resolve()),
        "adapters": {
            name: _describe_adapter(path)
            for name, path in ADAPTER_PATHS.items()
        },
    }


@app.post("/predict")
def predict_all(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(400, "text không được để trống")

    methods = req.methods or list(ADAPTER_PATHS.keys())
    results = {}

    for method in methods:
        if method not in ADAPTER_PATHS:
            results[method] = {"error": f"method không hợp lệ: {method}"}
            continue

        try:
            results[method] = _infer(req.text, method)
        except FileNotFoundError as e:
            results[method] = {"error": str(e)}
        except RuntimeError as e:
            results[method] = {"error": str(e)}
        except Exception as e:
            log.exception(f"[{method}] inference error")
            results[method] = {"error": f"{type(e).__name__}: {e}"}

    return {"text": req.text, "results": results, "mock": False}


@app.post("/predict/single")
def predict_single(req: SingleRequest):
    if req.method not in ADAPTER_PATHS:
        raise HTTPException(400, f"method phải là: {list(ADAPTER_PATHS.keys())}")

    return predict_all(PredictRequest(text=req.text, methods=[req.method]))


@app.post("/preload")
def preload_all():
    status = {}

    for method in ADAPTER_PATHS:
        try:
            _load_model(method)
            status[method] = "loaded"
        except Exception as e:
            status[method] = f"error: {e}"

    return status


@app.get("/examples")
def get_examples():
    return {
        "examples": [
            "This movie was absolutely fantastic! The acting and storyline were superb.",
            "Terrible waste of time. I fell asleep halfway through and never bothered finishing.",
            "An average film — nothing special, but not unwatchable either.",
            "One of the best performances I have ever seen. A masterpiece of modern cinema.",
            "The plot was confusing, the characters were boring, and the ending was disappointing.",
            "Surprisingly good! I had low expectations but ended up really enjoying every minute.",
            "Not bad, not great. A forgettable but harmless way to spend two hours.",
            "Absolutely dreadful. The worst film I have seen in years — avoid at all costs.",
        ]
    }

# cd C:\Users\ASUS\Documents\NLP\NHOM10_CUOIKY_NLP\TestFastAPI
# python -m uvicorn main:app --reload --port 8000
# http://127.0.0.1:8000/docs
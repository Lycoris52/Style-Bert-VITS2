import argparse
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

import style_bert_vits2.nlp.symbols as symbols
import style_bert_vits2.tts_model as tts_model
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
from style_bert_vits2.nlp.japanese.user_dict import update_dict

SAMPLE_RATE = 44100
HOP_SIZE = 512

VOWEL_MAP = {
    "a": "A",
    "i": "I",
    "u": "U",
    "e": "E",
    "o": "O",
}

REST_SYMBOLS = {
    ".", ",", "!", "?", "-", "…",
    "sil", "pau", "sp", "cl", "N",
}

BLANK_SYMBOLS = {"_"}

def build_aiueo_timeline(phonemes: list[str], dur_frames: list[int]) -> list[dict[str, Any]]:
    sec_per_frame = HOP_SIZE / SAMPLE_RATE
    t = 0.0
    events: list[dict[str, Any]] = []

    current_vowel = None
    current_start = None
    current_weight = 95

    def close_current(end_time: float) -> None:
        nonlocal current_vowel, current_start
        if current_vowel is not None and current_start is not None and end_time > current_start:
            events.append({
                "t0": round(current_start, 4),
                "t1": round(end_time, 4),
                "v": current_vowel,
                "w": current_weight,
            })
        current_vowel = None
        current_start = None

    for p, df in zip(phonemes, dur_frames):
        dt = int(df) * sec_per_frame
        t0 = t
        t1 = t + dt

        if p in BLANK_SYMBOLS:
            pass

        elif p in VOWEL_MAP:
            vowel = VOWEL_MAP[p]
            if current_vowel != vowel:
                close_current(t0)
                current_vowel = vowel
                current_start = t0

        elif p in REST_SYMBOLS:
            close_current(t0)
            events.append({
                "t0": round(t0, 4),
                "t1": round(t1, 4),
                "v": "REST",
                "w": 60,
            })

        t = t1
        
    close_current(t)
    return merge_same_events(events)

def merge_same_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return []

    merged = [events[0]]
    for e in events[1:]:
        prev = merged[-1]
        if prev["v"] == e["v"] and abs(prev["t1"] - e["t0"]) < 1e-6:
            prev["t1"] = e["t1"]
            prev["w"] = max(prev["w"], e["w"])
        else:
            merged.append(e)
    return merged

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Japanese text to synthesize")

class TTSResponse(BaseModel):
    utterance: str
    sample_rate: int
    blend: str
    audio_wave: list[float] # 本番ではこちらを有効にしてください。curl や docs のテスト画面だと表示しきれませんので。。。

class TTSServer:
    def __init__(self) -> None:
        # このプロセスからはワーカーを起動して辞書を使いたいので、ここで初期化
        pyopenjtalk_worker.initialize_worker()
        update_dict()

        self.model = tts_model.TTSModel(
            model_path=Path("model_assets/Kano-Yukishiro/Kano-Yukishiro.safetensors"),
            config_path=Path("model_assets/Kano-Yukishiro/config.json"),
            style_vec_path=Path("model_assets/Kano-Yukishiro/style_vectors.npy"),
            device="cuda",
        )

        # last_lipsync_info を使うので、同時推論でデータが混ざらないようにロック
        self._lock = threading.Lock()

    def tts(self, text: str) -> np.ndarray:
        sr, audio = self.model.infer(
            text=text,
            language="JP",
            reference_audio_path=None,
            sdp_ratio=0.2,
            noise=0.6,
            noise_w=0.8,
            length=1,
            line_split=False,
            split_interval=0.5,
            assist_text="",
            assist_text_weight=1,
            use_assist_text=False,
            style="Neutral",
            style_weight=5,
            given_tone=None,
            speaker_id=0,
            pitch_scale=1,
            intonation_scale=1,
        )

        # Unity AudioClip 用に int16 -> float32 [-1, 1]
        return audio.astype(np.float32, order="C") / 32768.0

    def generate_audio_with_timeline(self, text: str) -> dict[str, Any]:
        with self._lock:
            audio = self.tts(text)

            lipsync_info = getattr(self.model, "last_lipsync_info", None)
            if not lipsync_info:
                raise RuntimeError("model.last_lipsync_info is empty")

            phone_ids = lipsync_info.get("phone_ids")
            durations = lipsync_info.get("durations")

            if phone_ids is None or durations is None:
                raise RuntimeError("phone_ids or durations not found in model.last_lipsync_info")

            phoneme_dict = symbols.SYMBOLS
            phonemes = [phoneme_dict[val] for val in phone_ids]

            timeline = build_aiueo_timeline(phonemes, durations)
            blend = {"events": timeline}

            return {
                "utterance": text,
                "sample_rate": SAMPLE_RATE,
                "blend": json.dumps(blend),
                "audio_wave": audio.tolist(), # 本番ではこちらを有効にしてください。curl や docs のテスト画面だと表示しきれませんので。。。
            }

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html

def custom_swagger_html(openapi_url: str, title: str) -> str:
    html = get_swagger_ui_html(
        openapi_url=openapi_url,
        title=title,
        swagger_ui_parameters={
            "syntaxHighlight": False,
            "defaultModelsExpandDepth": -1,
            "defaultModelExpandDepth": 0,
            "docExpansion": "none",
            "displayRequestDuration": True,
        },
    ).body.decode("utf-8")

    inject_script = r"""
<script>
(function () {
    const MAX_LEN = 4000;
    const CHECK_INTERVAL_MS = 800;

    function truncateLongResponses() {
        const candidates = document.querySelectorAll('pre, code');

        for (const el of candidates) {
            if (!el || !el.textContent) continue;

            const text = el.textContent;
            if (text.length <= MAX_LEN) continue;
            if (el.dataset.truncatedByCustomDocs === "1") continue;

            const preview = text.slice(0, MAX_LEN);
            el.textContent =
                preview +
                "\n\n... [truncated in docs UI] ..." +
                "\nOriginal response length: " + text.length + " characters";

            el.dataset.truncatedByCustomDocs = "1";
            el.style.whiteSpace = "pre-wrap";
            el.style.wordBreak = "break-word";
            el.style.maxHeight = "480px";
            el.style.overflow = "auto";
        }
    }

    function install() {
        truncateLongResponses();

        const observer = new MutationObserver(function () {
            truncateLongResponses();
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        setInterval(truncateLongResponses, CHECK_INTERVAL_MS);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", install);
    } else {
        install();
    }
})();
</script>
"""
    return html.replace("</body>", inject_script + "\n</body>")

def docs_openapi_url(app: FastAPI) -> str:
    root = (app.root_path or "").rstrip("/")
    openapi = app.openapi_url or "/openapi.json"
    if root:
        return f"{root}{openapi}"
    return openapi
    
def create_app(server: TTSServer, root_path: str = "") -> FastAPI:
    app = FastAPI(
        title="Style-Bert-VITS2 TTS API",
        version="1.0.0",
        root_path=root_path,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # ローカルでやる場合、CORS 許可は要らない
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        GZipMiddleware,
        minimum_size=500,
        compresslevel=5,
    )

    @app.get("/docs-lite", include_in_schema=False)
    def overridden_swagger():
        return HTMLResponse(
            custom_swagger_html(
                openapi_url=docs_openapi_url(app),
                title=app.title + " - Docs Lite",
            )
        )

    @app.post("/tts", response_model=TTSResponse)
    def tts_endpoint(req: TTSRequest):
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text must not be empty")

        try:
            return server.generate_audio_with_timeline(text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Style-Bert-VITS2 FastAPI server")
    parser.add_argument("--server_name", default="127.0.0.1", help="Host to bind, e.g. 0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7865, help="Port to bind")
    parser.add_argument("--root_path", default="", help="ASGI root_path, e.g. /fastapi")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    server = TTSServer()
    app = create_app(server, root_path=args.root_path)

    uvicorn.run(
        app,
        host=args.server_name,
        port=args.server_port,
    )

if __name__ == "__main__":
    main()
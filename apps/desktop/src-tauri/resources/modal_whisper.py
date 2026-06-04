import contextlib
import math
import queue
import threading
import wave
import json

from modal import App, Image, fastapi_endpoint, enter, method

try:
    from fastapi import Request
    from fastapi.responses import StreamingResponse
except ImportError:
    pass

app = App("crisper-whisper-app")

image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "git+https://github.com/nyrahealth/transformers.git@crisper_whisper",
        "accelerate>=0.28.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "fastapi",
        "python-multipart"
    )
    .run_commands(
        'python -c "from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; model_id = \'nyrahealth/CrisperWhisper\'; AutoModelForSpeechSeq2Seq.from_pretrained(model_id); AutoProcessor.from_pretrained(model_id)"'
    )
)

def get_audio_duration(file_path):
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)
    except Exception:
        return 0

def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]

        if current_end is None or next_start is None:
            continue

        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)
            if next_end is not None:
                adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
            else:
                adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, None)

    pipeline_output["chunks"] = adjusted_chunks
    return pipeline_output


@app.cls(
    gpu="A100-80GB",
    timeout=600,
    image=image,
    env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
)
class CrisperWhisper:
    @enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32
        model_id = "nyrahealth/CrisperWhisper"
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="eager"
        )
        model.to(self.device)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=4 if self.device == "cuda:0" else 1,
            return_timestamps="word",
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    @method(is_generator=True)
    def transcribe(self, audio_bytes: bytes):
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
            
        orig_forward = self.pipe.forward
        try:
            duration = get_audio_duration(tmp_path)
            step = 20
            total_chunks = max(1, math.ceil((duration - 30) / step) + 1) if duration > 30 else 1

            q = queue.Queue()
            chunks_done = 0

            def patched_forward(*args_fw, **kwargs_fw):
                nonlocal chunks_done
                chunks_done += 1
                progress = min(99, int((chunks_done / total_chunks) * 100))
                q.put({"progress": progress})
                return orig_forward(*args_fw, **kwargs_fw)

            self.pipe.forward = patched_forward

            def run_pipeline():
                try:
                    import torch
                    with torch.no_grad():
                        result = self.pipe(tmp_path)
                    result = adjust_pauses_for_hf_pipeline_output(result)
                    q.put({"result": result})
                except Exception as e:
                    q.put({"error": str(e)})

            t = threading.Thread(target=run_pipeline)
            t.start()

            while True:
                msg = q.get()
                yield msg
                if "result" in msg or "error" in msg:
                    break

            t.join()
        finally:
            self.pipe.forward = orig_forward
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@app.function(image=image)
@fastapi_endpoint(method="POST")
async def transcribe_endpoint(request: 'Request'):
    audio_bytes = await request.body()
    
    def generate():
        model = CrisperWhisper()
        for msg in model.transcribe.remote_gen(audio_bytes):
            yield json.dumps(msg) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

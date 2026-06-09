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
        
        # Monkeypatch Whisper's token timestamp extraction to prevent crash on empty/silent segments
        # (e.g., shape-0 tensor issues during DTW or median filtering, see nyrahealth/CrisperWhisper#43)
        from transformers.models.whisper.generation_whisper import WhisperGenerationMixin, _median_filter, _dynamic_time_warping2

        def patched_extract_token_timestamps(self, generate_outputs, alignment_heads, time_precision=0.02, num_frames=None):
            import torch
            import numpy as np

            cross_attentions = []
            for i in range(self.config.decoder_layers):
                cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))

            weights = torch.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
            weights = weights.permute([1, 0, 2, 3])

            if "beam_indices" in generate_outputs:
                weight_length = (generate_outputs.beam_indices != -1).sum(-1).max()
                weights = weights[:, :, :weight_length]

                beam_indices = generate_outputs.beam_indices[:, :weight_length]
                beam_indices = beam_indices.masked_fill(beam_indices == -1, 0)

                weights = torch.stack(
                    [
                        torch.index_select(weights[:, :, i, :], dim=0, index=beam_indices[:, i])
                        for i in range(beam_indices.shape[1])
                    ],
                    dim=2,
                )

            timestamps = torch.zeros_like(generate_outputs.sequences, dtype=torch.float32)
            batch_size = timestamps.shape[0]

            if num_frames is not None:
                if len(np.unique(num_frames)) == 1:
                    num_frames = num_frames if isinstance(num_frames, int) else num_frames[0]
                    weights = weights[..., : num_frames // 2]
                else:
                    repeat_time = batch_size if isinstance(num_frames, int) else batch_size // len(num_frames)
                    num_frames = np.repeat(num_frames, repeat_time)

            if num_frames is None or isinstance(num_frames, int):
                if self.generation_config.legacy:
                    std = torch.std(weights, dim=-2, keepdim=True, unbiased=False)
                    mean = torch.mean(weights, dim=-2, keepdim=True)
                    weights = (weights - mean) / std
                weights = _median_filter(weights, self.generation_config.median_filter_width)
                weights = weights.mean(dim=1)

            for batch_idx in range(batch_size):
                non_special_tokens_indices = []
                special_tokens_indices = []
                for index_, token_id in enumerate(generate_outputs.sequences[batch_idx, 1:]):
                    if self.generation_config.token_ids_to_ignore_for_dtw:
                        if token_id not in self.generation_config.token_ids_to_ignore_for_dtw and token_id < self.model.config.eos_token_id:
                            non_special_tokens_indices.append(index_)
                        else:
                            special_tokens_indices.append(index_)

                if not non_special_tokens_indices:
                    continue

                if num_frames is not None and isinstance(num_frames, (tuple, list, np.ndarray)):
                    matrix = weights[batch_idx, :, non_special_tokens_indices, :num_frames[batch_idx] // 2]

                    if self.generation_config.legacy:
                        std = torch.std(matrix, dim=-2, keepdim=True, unbiased=False)
                        mean = torch.mean(matrix, dim=-2, keepdim=True)
                        matrix = (matrix - mean) / std
                    matrix = _median_filter(matrix, self.generation_config.median_filter_width)
                    matrix = matrix.mean(dim=0)
                else:
                    matrix = weights[batch_idx, non_special_tokens_indices, :]

                text_indices, time_indices = _dynamic_time_warping2(-matrix.cpu().double().numpy(), allow_vertical_moves=False)
                jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
                jump_times = list((time_indices[jumps] * time_precision).round(4))
                for index_ in special_tokens_indices:
                    next_element = jump_times[index_] if index_ < len(jump_times) else round((time_indices[-1] * time_precision), 4)
                    jump_times.insert(index_, next_element)
                timestamps[batch_idx, 1:] = torch.tensor(jump_times)

            return timestamps

        WhisperGenerationMixin._extract_token_timestamps = patched_extract_token_timestamps

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

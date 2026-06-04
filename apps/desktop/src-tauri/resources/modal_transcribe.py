import os
import sys
import json
import modal

app = modal.App("crisper-whisper-runner")

@app.local_entrypoint()
def main(audio_path: str, model_type: str = "crisper-whisper"):
	if not os.path.exists(audio_path):
		print(json.dumps({"error": f"Audio file not found: {audio_path}"}))
		sys.exit(1)

	with open(audio_path, "rb") as f:
		audio_bytes = f.read()

	try:
		Cls = modal.Cls.from_name("crisper-whisper-app", "CrisperWhisper")
		instance = Cls()
		result_msg = None
		for msg in instance.transcribe.remote_gen(audio_bytes):
			if "result" in msg:
				result_msg = msg["result"]
				break
			elif "error" in msg:
				print(json.dumps({"error": msg["error"]}))
				sys.exit(1)

		if not result_msg or "chunks" not in result_msg:
			print(json.dumps({"error": "No transcription result returned from CrisperWhisper"}))
			sys.exit(1)

		chunks = result_msg["chunks"]
		words = []
		for chunk in chunks:
			text = chunk.get("text", "")
			timestamp = chunk.get("timestamp")
			if timestamp is not None and len(timestamp) >= 2:
				raw_start = timestamp[0] if timestamp[0] is not None else 0.0
				raw_end = timestamp[1] if timestamp[1] is not None else (raw_start + 0.5)
			else:
				raw_start = 0.0
				raw_end = 0.0

			start = max(0.0, raw_start - 0.15)
			end = raw_end

			words.append({
				"text": str(text).strip(),
				"start": float(start),
				"end": float(end)
			})

		segments = []
		max_words = 6
		for chunk_idx, i in enumerate(range(0, len(words), max_words)):
			chunk = words[i:i + max_words]
			segment_text = " ".join([w["text"] for w in chunk])
			segment_start = chunk[0]["start"] if chunk else 0.0
			segment_end = chunk[-1]["end"] if chunk else 0.0

			segments.append({
				"id": f"segment-{chunk_idx}",
				"start": segment_start,
				"end": segment_end,
				"text": segment_text,
				"words": chunk
			})

		print(json.dumps({"segments": segments}))
	except Exception as e:
		print(json.dumps({"error": f"CrisperWhisper error: {str(e)}"}))
		sys.exit(1)

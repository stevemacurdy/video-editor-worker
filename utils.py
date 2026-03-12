"""
WoulfAI Video Editor Worker — Utility Functions
All FFmpeg, Whisper, fuzzy matching, and Supabase helpers.
"""

import os
import re
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import httpx
from openai import OpenAI
from fuzzywuzzy import fuzz
from supabase import create_client

# ── Clients ───────────────────────────────────────────────────────────────────

_openai_client: Optional[OpenAI] = None
_supabase_client = None


def get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _openai_client


def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"],
        )
    return _supabase_client


# ── Download ──────────────────────────────────────────────────────────────────

async def download_video(url: str, dest: str):
    """Stream-download a video from URL to local path."""
    async with httpx.AsyncClient(timeout=1800, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=1024 * 64):
                    f.write(chunk)
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        raise RuntimeError(f"Downloaded file is empty or missing: {dest}")


# ── FFprobe / Duration ────────────────────────────────────────────────────────

def get_duration(path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", path,
        ],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return 0.0


# ── Thumbnail ─────────────────────────────────────────────────────────────────

def generate_thumbnail(video_path: str, output_path: str, timestamp: float = 1.0):
    """Extract a single JPEG frame from a video."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(min(timestamp, max(get_duration(video_path) - 0.5, 0))),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
        ],
        capture_output=True, check=True,
    )


# ── Supabase Upload ──────────────────────────────────────────────────────────

async def upload_to_supabase(local_path: str, bucket: str, remote_path: str) -> str:
    """Upload a file to Supabase Storage and return the public URL."""
    sb = get_supabase()
    with open(local_path, "rb") as f:
        data = f.read()

    # Determine content type
    ext = Path(local_path).suffix.lower()
    content_types = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    content_type = content_types.get(ext, "application/octet-stream")

    sb.storage.from_(bucket).upload(
        remote_path,
        data,
        file_options={"content-type": content_type, "upsert": "true"},
    )

    url_data = sb.storage.from_(bucket).get_public_url(remote_path)
    return url_data


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe_video(video_path: str) -> dict:
    """Transcribe video audio with OpenAI Whisper (word + segment granularity)."""
    client = get_openai()

    # Extract audio to a smaller file for Whisper (max 25MB)
    audio_path = video_path + ".audio.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "libmp3lame", "-b:a", "64k",
            "-ac", "1", "-ar", "16000",
            audio_path,
        ],
        capture_output=True, check=True,
    )

    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

    # Clean up temp audio
    try:
        os.remove(audio_path)
    except OSError:
        pass

    segments = []
    if hasattr(result, "segments") and result.segments:
        for s in result.segments:
            segments.append({
                "start": getattr(s, "start", 0),
                "end": getattr(s, "end", 0),
                "text": getattr(s, "text", "").strip(),
            })

    words = []
    if hasattr(result, "words") and result.words:
        for w in result.words:
            words.append({
                "start": getattr(w, "start", 0),
                "end": getattr(w, "end", 0),
                "word": getattr(w, "word", ""),
            })

    return {
        "text": getattr(result, "text", ""),
        "segments": segments,
        "words": words,
    }


# ── Fuzzy Quote Matching ─────────────────────────────────────────────────────

def find_quotes(transcript: dict, quotes: list[str], threshold: int = 70) -> list[dict]:
    """Fuzzy-match user quotes against transcript segments."""
    segments = transcript.get("segments", [])
    if not segments:
        return []

    matches = []
    for quote in quotes:
        quote_lower = quote.lower().strip()
        if not quote_lower:
            continue

        best_score = 0
        best_match = None

        # Try individual segments
        for seg in segments:
            score = fuzz.partial_ratio(quote_lower, seg["text"].lower())
            if score > best_score:
                best_score = score
                best_match = seg

        # Try sliding windows of 2–3 consecutive segments for longer quotes
        for i in range(len(segments) - 1):
            combined_2 = segments[i]["text"] + " " + segments[i + 1]["text"]
            score = fuzz.partial_ratio(quote_lower, combined_2.lower())
            if score > best_score:
                best_score = score
                best_match = {
                    "start": segments[i]["start"],
                    "end": segments[i + 1]["end"],
                    "text": combined_2,
                }
            if i < len(segments) - 2:
                combined_3 = combined_2 + " " + segments[i + 2]["text"]
                score = fuzz.partial_ratio(quote_lower, combined_3.lower())
                if score > best_score:
                    best_score = score
                    best_match = {
                        "start": segments[i]["start"],
                        "end": segments[i + 2]["end"],
                        "text": combined_3,
                    }

        if best_match and best_score >= threshold:
            matches.append({
                "quote": quote,
                "matched_text": best_match["text"],
                "start": max(0, best_match["start"] - 0.5),
                "end": best_match["end"] + 0.5,
                "confidence": best_score / 100.0,
            })

    return matches


# ── Power Clip Scoring ────────────────────────────────────────────────────────

POWER_WORDS = [
    "amazing", "incredible", "transform", "save", "boost", "increase",
    "reduce", "grow", "revenue", "profit", "results", "success",
    "million", "percent", "guarantee", "free", "new", "best",
    "proven", "exactly", "important", "critical", "game-changer",
]


def score_segments_for_power_clips(
    segments: list[dict], clip_min: int = 5, clip_max: int = 30
) -> list[dict]:
    """Score transcript segments and select the top non-overlapping clips."""
    scored = []

    for i, seg in enumerate(segments):
        text = seg.get("text", "")
        text_lower = text.lower()
        score = 0

        # Power words: +2 each
        for w in POWER_WORDS:
            if w in text_lower:
                score += 2

        # Starts with capital letter (sentence start): +1
        stripped = text.strip()
        if stripped and stripped[0].isupper():
            score += 1

        # Ends with punctuation (complete thought): +1
        if stripped and stripped[-1] in ".!?":
            score += 1

        # Word count sweet spot (10–40): +2
        word_count = len(text_lower.split())
        if 10 <= word_count <= 40:
            score += 2

        # Contains numbers (specific claims): +2
        if re.search(r"\d+", text):
            score += 2

        # Question mark (engagement): +1
        if "?" in text:
            score += 1

        # Direct address ("you" / "your"): +1
        if "you" in text_lower.split() or "your" in text_lower.split():
            score += 1

        scored.append({
            **seg,
            "score": score,
            "index": i,
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Select top clips, expanding short ones and ensuring no overlap
    clips = []
    used_ranges: list[tuple[float, float]] = []

    for seg in scored:
        if len(clips) >= 8:
            break

        start = seg.get("start", 0)
        end = seg.get("end", 0)
        duration = end - start

        # Expand short clips by including adjacent segments
        idx = seg["index"]
        if duration < clip_min:
            right_idx = idx
            while end - start < clip_min and right_idx + 1 < len(segments):
                right_idx += 1
                end = segments[right_idx]["end"]
            left_idx = idx
            while end - start < clip_min and left_idx > 0:
                left_idx -= 1
                start = segments[left_idx]["start"]

        # Truncate if too long
        if end - start > clip_max:
            end = start + clip_max

        # Skip if still too short
        if end - start < clip_min:
            continue

        # Check overlap with already-selected clips
        overlap = False
        for us, ue in used_ranges:
            if start < ue and end > us:
                overlap = True
                break

        if not overlap:
            clips.append({
                "start": start,
                "end": end,
                "text": seg["text"],
                "score": seg["score"],
            })
            used_ranges.append((start, end))

    return clips


# ── FFmpeg Clip Extraction ────────────────────────────────────────────────────

def extract_clip(
    video_path: str, start: float, end: float, output_path: str
):
    """Extract a clip using FFmpeg with re-encoding for frame-accurate cuts."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start),
            "-to", str(end),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ],
        capture_output=True, check=True,
    )


# ── Captions + Format Crop ────────────────────────────────────────────────────

def add_captions_and_crop(
    video_path: str, output_path: str, text: str, fmt: str = "16:9"
):
    """Burn captions into video and crop/resize for the target format."""
    # Escape special characters for FFmpeg drawtext
    escaped = (
        text
        .replace("\\", "\\\\")
        .replace("'", "\u2019")   # smart quote to avoid shell issues
        .replace(":", "\\:")
        .replace('"', '\\"')
    )

    fontsize = 24 if fmt == "16:9" else 20

    # Build the drawtext + crop/scale filter chain
    drawtext = (
        f"drawtext=text='{escaped}'"
        f":fontsize={fontsize}"
        f":fontcolor=white:borderw=2:bordercolor=black"
        f":x=(w-text_w)/2:y=h-th-40"
    )

    if fmt == "9:16":
        vf = f"{drawtext},crop=ih*9/16:ih,scale=1080:1920"
    elif fmt == "1:1":
        vf = f"{drawtext},crop=min(iw\\,ih):min(iw\\,ih),scale=1080:1080"
    else:  # 16:9
        vf = (
            f"scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,{drawtext}"
        )

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ],
        capture_output=True, check=True,
    )


# ── Video Cleanup ─────────────────────────────────────────────────────────────

def cleanup_video(video_path: str, output_path: str, options: dict):
    """
    Professional cleanup: audio normalization, noise reduction,
    color correction, and optional stabilization.
    """
    vfilters: list[str] = []
    afilters: list[str] = []

    if options.get("normalize_audio", True):
        afilters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    if options.get("reduce_noise", True):
        afilters.append("highpass=f=80")
        afilters.append("lowpass=f=12000")

    if options.get("color_correct", True):
        vfilters.append("eq=contrast=1.05:brightness=0.02:saturation=1.1")

    if options.get("stabilize", False):
        # Two-pass stabilization
        detect_path = output_path + ".trf"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={detect_path}",
                "-f", "null", "-",
            ],
            capture_output=True, check=True,
        )
        vfilters.append(f"vidstabtransform=input={detect_path}:smoothing=10")

    cmd = ["ffmpeg", "-y", "-i", video_path]
    if vfilters:
        cmd += ["-vf", ",".join(vfilters)]
    if afilters:
        cmd += ["-af", ",".join(afilters)]

    cmd += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]

    subprocess.run(cmd, capture_output=True, check=True)


# ── Dead Air Removal ──────────────────────────────────────────────────────────

def remove_dead_air(
    video_path: str, output_path: str, silence_threshold: float = -30, min_gap: float = 2.0
):
    """
    Detect silence gaps > min_gap seconds, then concat the non-silent segments.
    Uses FFmpeg silencedetect + concat protocol.
    """
    # Step 1: Detect silence
    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-af", f"silencedetect=noise={silence_threshold}dB:d={min_gap}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )

    stderr = result.stderr
    silence_starts: list[float] = []
    silence_ends: list[float] = []

    for line in stderr.split("\n"):
        if "silence_start:" in line:
            try:
                val = float(line.split("silence_start:")[1].strip().split()[0])
                silence_starts.append(val)
            except (ValueError, IndexError):
                pass
        if "silence_end:" in line:
            try:
                val = float(line.split("silence_end:")[1].strip().split()[0])
                silence_ends.append(val)
            except (ValueError, IndexError):
                pass

    # If no silence detected, just copy
    if not silence_starts:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-c", "copy", output_path],
            capture_output=True, check=True,
        )
        return

    # Build non-silent segments
    duration = get_duration(video_path)
    segments: list[tuple[float, float]] = []
    pos = 0.0

    for i, s_start in enumerate(silence_starts):
        if s_start > pos:
            segments.append((pos, s_start))
        if i < len(silence_ends):
            pos = silence_ends[i]
        else:
            pos = duration

    # Remaining segment after last silence
    if pos < duration:
        segments.append((pos, duration))

    if not segments:
        # All silence, just copy original
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-c", "copy", output_path],
            capture_output=True, check=True,
        )
        return

    # Step 2: Extract each segment and concat
    tmpdir = os.path.dirname(output_path)
    seg_paths: list[str] = []
    concat_lines: list[str] = []

    for i, (seg_start, seg_end) in enumerate(segments):
        seg_path = os.path.join(tmpdir, f"_seg_{i:04d}.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", str(seg_start),
                "-to", str(seg_end),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                seg_path,
            ],
            capture_output=True, check=True,
        )
        seg_paths.append(seg_path)
        concat_lines.append(f"file '{seg_path}'")

    concat_file = os.path.join(tmpdir, "_concat.txt")
    with open(concat_file, "w") as f:
        f.write("\n".join(concat_lines))

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ],
        capture_output=True, check=True,
    )

    # Cleanup temp files
    for p in seg_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.remove(concat_file)
    except OSError:
        pass


# ── Stitch Clips ──────────────────────────────────────────────────────────────

def stitch_clips(clip_paths: list[str], output_path: str):
    """Concatenate multiple video clips into one compilation."""
    tmpdir = os.path.dirname(output_path) or tempfile.gettempdir()
    concat_file = os.path.join(tmpdir, "_stitch_concat.txt")

    with open(concat_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ],
        capture_output=True, check=True,
    )

    try:
        os.remove(concat_file)
    except OSError:
        pass


# ── Supabase Job Updates ─────────────────────────────────────────────────────

async def update_job_status(job_id: str, status: str, error: str = ""):
    """Update a video_jobs row status (and optional error) in Supabase."""
    sb = get_supabase()
    update: dict = {"status": status}
    if error:
        update["error"] = error
    sb.table("video_jobs").update(update).eq("id", job_id).execute()


async def update_job_transcript(job_id: str, transcript: dict):
    """Save the transcript JSON to the video_jobs row."""
    sb = get_supabase()
    sb.table("video_jobs").update({"transcript": transcript}).eq("id", job_id).execute()


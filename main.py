"""
WoulfAI Video Editor Worker
FastAPI application deployed on Railway.
Handles video transcription, quote extraction, power clip generation, and cleanup.
"""

import os
import json
import time
import tempfile
import traceback
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from utils import (
    download_video,
    get_duration,
    generate_thumbnail,
    upload_to_supabase,
    stitch_clips,
    transcribe_video,
    find_quotes,
    score_segments_for_power_clips,
    extract_clip,
    add_captions_and_crop,
    cleanup_video,
    remove_dead_air,
    update_job_status,
    update_job_transcript,
)

app = FastAPI(title="WoulfAI Video Editor Worker", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKER_SECRET = os.environ.get("WORKER_SECRET", "")


def verify_auth(authorization: Optional[str] = None):
    if not WORKER_SECRET:
        return  # No secret configured, skip auth
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization")
    token = authorization.replace("Bearer ", "")
    if token != WORKER_SECRET:
        raise HTTPException(status_code=401, detail="Invalid authorization")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    import subprocess
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        ffmpeg_ok = True
    except Exception:
        ffmpeg_ok = False
    return {"status": "ok", "ffmpeg": ffmpeg_ok}


# ── Models ────────────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    jobId: str
    mode: str  # quote | power | cleanup
    sourceUrl: str
    quotes: list[str] = []
    clipMin: int = 5
    clipMax: int = 30
    clipFormats: list[str] = ["16:9"]
    cleanupOptions: dict = {}
    stitch: bool = False
    callbackUrl: str = ""


class TranscribeRequest(BaseModel):
    sourceUrl: str


# ── Transcribe Only ──────────────────────────────────────────────────────────

@app.post("/transcribe-only")
async def transcribe_only(req: TranscribeRequest, authorization: Optional[str] = Header(None)):
    verify_auth(authorization)

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "input.mp4")
        await download_video(req.sourceUrl, video_path)
        transcript = transcribe_video(video_path)
        return {"transcript": transcript}


# ── Main Process ──────────────────────────────────────────────────────────────

@app.post("/process")
async def process(req: ProcessRequest, authorization: Optional[str] = Header(None)):
    verify_auth(authorization)

    # Run processing in background (fire and forget pattern)
    import asyncio
    asyncio.create_task(_process_job(req))
    return {"status": "accepted", "jobId": req.jobId}


async def _process_job(req: ProcessRequest):
    """Main processing pipeline. Runs in background."""
    start_time = time.time()
    clips_result = []

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")

            # Step 1: Update status to transcribing
            await update_job_status(req.jobId, "transcribing")

            # Step 2: Download video
            await download_video(req.sourceUrl, video_path)
            duration = get_duration(video_path)

            # Step 3: Transcribe
            transcript = transcribe_video(video_path)
            await update_job_transcript(req.jobId, transcript)

            # Step 4: Update status to processing
            await update_job_status(req.jobId, "processing")

            # Step 5: Mode-specific logic
            if req.mode == "quote":
                clips_result = await _process_quote_mode(
                    video_path, tmpdir, req.jobId, transcript, req.quotes, req.stitch
                )
            elif req.mode == "power":
                clips_result = await _process_power_mode(
                    video_path, tmpdir, req.jobId, transcript,
                    req.clipMin, req.clipMax, req.clipFormats
                )
            elif req.mode == "cleanup":
                clips_result = await _process_cleanup_mode(
                    video_path, tmpdir, req.jobId, req.cleanupOptions
                )
            else:
                raise ValueError(f"Unknown mode: {req.mode}")

        # Step 6: Callback with results
        processing_seconds = int(time.time() - start_time)
        await _send_callback(req.callbackUrl, {
            "jobId": req.jobId,
            "status": "complete",
            "clips": clips_result,
            "processingSeconds": processing_seconds,
        })

    except Exception as e:
        error_msg = str(e)
        print(f"Job {req.jobId} failed: {error_msg}")
        traceback.print_exc()

        await update_job_status(req.jobId, "failed", error=error_msg)
        await _send_callback(req.callbackUrl, {
            "jobId": req.jobId,
            "status": "failed",
            "error": error_msg,
        })


# ── Quote Mode ────────────────────────────────────────────────────────────────

async def _process_quote_mode(
    video_path: str, tmpdir: str, job_id: str,
    transcript: dict, quotes: list[str], do_stitch: bool
) -> list[dict]:
    matches = find_quotes(transcript, quotes)
    if not matches:
        return []

    clips = []
    clip_paths = []

    for i, match in enumerate(matches):
        clip_filename = f"clip_{i:03d}.mp4"
        clip_path = os.path.join(tmpdir, clip_filename)
        thumb_path = os.path.join(tmpdir, f"thumb_{i:03d}.jpg")

        extract_clip(video_path, match["start"], match["end"], clip_path)
        generate_thumbnail(clip_path, thumb_path)

        # Upload clip and thumbnail
        clip_url = await upload_to_supabase(clip_path, "video-outputs", f"{job_id}/{clip_filename}")
        thumb_url = await upload_to_supabase(thumb_path, "video-outputs", f"{job_id}/thumb_{i:03d}.jpg")

        clip_paths.append(clip_path)
        clips.append({
            "start": match["start"],
            "end": match["end"],
            "duration": match["end"] - match["start"],
            "matchedQuote": match["quote"],
            "transcriptSegment": match["matched_text"],
            "confidence": match["confidence"],
            "downloadUrl": clip_url,
            "thumbnailUrl": thumb_url,
            "format": "16:9",
            "captionsBurned": False,
        })

    # Optionally stitch all clips together
    if do_stitch and len(clip_paths) > 1:
        stitched_path = os.path.join(tmpdir, "compilation.mp4")
        stitch_clips(clip_paths, stitched_path)
        stitch_url = await upload_to_supabase(
            stitched_path, "video-outputs", f"{job_id}/compilation.mp4"
        )
        clips.append({
            "start": 0,
            "end": 0,
            "duration": 0,
            "matchedQuote": "Compilation",
            "transcriptSegment": f"All {len(clip_paths)} clips stitched together",
            "confidence": 1.0,
            "downloadUrl": stitch_url,
            "thumbnailUrl": clips[0]["thumbnailUrl"] if clips else None,
            "format": "16:9",
            "captionsBurned": False,
        })

    return clips


# ── Power Clips Mode ──────────────────────────────────────────────────────────

async def _process_power_mode(
    video_path: str, tmpdir: str, job_id: str,
    transcript: dict, clip_min: int, clip_max: int, clip_formats: list[str]
) -> list[dict]:
    segments = transcript.get("segments", [])
    if not segments:
        return []

    scored_clips = score_segments_for_power_clips(segments, clip_min, clip_max)
    if not scored_clips:
        return []

    results = []

    for i, sc in enumerate(scored_clips[:8]):
        base_clip = os.path.join(tmpdir, f"power_{i:03d}_base.mp4")
        extract_clip(video_path, sc["start"], sc["end"], base_clip)

        for fmt in clip_formats:
            out_filename = f"power_{i:03d}_{fmt.replace(':', 'x')}.mp4"
            out_path = os.path.join(tmpdir, out_filename)
            thumb_path = os.path.join(tmpdir, f"thumb_power_{i:03d}_{fmt.replace(':', 'x')}.jpg")

            # Add captions and crop to format
            add_captions_and_crop(base_clip, out_path, sc["text"], fmt)
            generate_thumbnail(out_path, thumb_path)

            clip_url = await upload_to_supabase(out_path, "video-outputs", f"{job_id}/{out_filename}")
            thumb_url = await upload_to_supabase(thumb_path, "video-outputs", f"{job_id}/thumb_power_{i:03d}_{fmt.replace(':', 'x')}.jpg")

            results.append({
                "start": sc["start"],
                "end": sc["end"],
                "duration": sc["end"] - sc["start"],
                "matchedQuote": None,
                "transcriptSegment": sc["text"],
                "confidence": sc["score"] / 20.0,  # Normalize to 0-1
                "downloadUrl": clip_url,
                "thumbnailUrl": thumb_url,
                "format": fmt,
                "captionsBurned": True,
            })

    return results


# ── Cleanup Mode ──────────────────────────────────────────────────────────────

async def _process_cleanup_mode(
    video_path: str, tmpdir: str, job_id: str, options: dict
) -> list[dict]:
    intermediate = video_path
    step = 0

    # Remove dead air first if enabled (default true since it's not in options but implied)
    if options.get("remove_dead_air", True):
        step += 1
        trimmed_path = os.path.join(tmpdir, f"step{step}_trimmed.mp4")
        remove_dead_air(intermediate, trimmed_path)
        if os.path.exists(trimmed_path) and os.path.getsize(trimmed_path) > 0:
            intermediate = trimmed_path

    # Main cleanup pass
    step += 1
    output_path = os.path.join(tmpdir, f"cleaned_output.mp4")
    cleanup_video(intermediate, output_path, options)

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Cleanup produced empty output")

    output_url = await upload_to_supabase(output_path, "video-outputs", f"{job_id}/cleaned.mp4")

    # Update job with output_url directly via Supabase
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    sb.table("video_jobs").update({"output_url": output_url}).eq("id", job_id).execute()

    return [{
        "start": 0,
        "end": 0,
        "duration": get_duration(output_path),
        "matchedQuote": None,
        "transcriptSegment": "Cleaned video",
        "confidence": 1.0,
        "downloadUrl": output_url,
        "thumbnailUrl": None,
        "format": "16:9",
        "captionsBurned": False,
    }]


# ── Callback Helper ──────────────────────────────────────────────────────────

async def _send_callback(url: str, payload: dict):
    if not url:
        return
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {WORKER_SECRET}",
                },
            )
    except Exception as e:
        print(f"Callback failed: {e}")


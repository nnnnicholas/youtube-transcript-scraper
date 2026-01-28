#!/usr/bin/env python3
"""
YouTube Channel Transcript Scraper

Scrapes all video transcripts from a YouTube channel using yt-dlp.
Supports auto-generated and manual captions, with resume capability.

Usage:
    python scrape_channel.py <channel_url_or_id> [--output-dir OUTPUT_DIR] [--lang LANG] [--resume]

Example:
    python scrape_channel.py https://www.youtube.com/@channelname --output-dir ./transcripts
    python scrape_channel.py UCxxxxxxxxxx --lang en --resume
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Concurrency/rate-limit defaults
DEFAULT_MAX_WORKERS = 3
MAX_RETRIES = 3
RATE_LIMIT_BACKOFF_BASE = 10.0  # seconds
RATE_LIMIT_BACKOFF_MAX = 120.0  # seconds
MIN_VIDEO_DURATION = 240  # seconds (skip videos shorter than this)


@dataclass
class VideoInfo:
    """Information about a single video."""
    id: str
    title: str
    url: str
    upload_date: Optional[str] = None
    duration: Optional[int] = None
    description: Optional[str] = None
    transcript_status: str = "pending"  # pending, success, no_transcript, error
    error_message: Optional[str] = None


@dataclass
class ChannelInfo:
    """Information about the channel."""
    id: str
    name: str
    url: str
    video_count: int
    scraped_at: str


def ensure_yt_dlp_installed() -> bool:
    """Check if yt-dlp is installed, install if not."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"yt-dlp version: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    logger.info("Installing yt-dlp...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", "yt-dlp"],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install yt-dlp: {e}")
        return False


def normalize_channel_url(channel_input: str, videos_only: bool = True) -> str:
    """Convert various channel input formats to a consistent URL."""
    # Already a full URL
    if channel_input.startswith("http"):
        if not videos_only:
            return channel_input

        parsed = urlparse(channel_input)
        path = parsed.path.rstrip("/")

        # Don't modify explicit subpages or non-channel URLs
        if any(seg in path for seg in ["/videos", "/shorts", "/streams", "/playlists", "/featured", "/community", "/live", "/releases"]):
            return channel_input
        if "/watch" in path or "/playlist" in path or "list=" in parsed.query:
            return channel_input

        new_path = f"{path}/videos" if path else "/videos"
        return urlunparse(parsed._replace(path=new_path))
    
    # Channel ID (starts with UC)
    if channel_input.startswith("UC") and len(channel_input) == 24:
        suffix = "/videos" if videos_only else ""
        return f"https://www.youtube.com/channel/{channel_input}{suffix}"
    
    # Handle @username format
    if channel_input.startswith("@"):
        suffix = "/videos" if videos_only else ""
        return f"https://www.youtube.com/{channel_input}{suffix}"
    
    # Assume it's a channel name/handle
    suffix = "/videos" if videos_only else ""
    return f"https://www.youtube.com/@{channel_input}{suffix}"


def infer_channel_name_from_url(channel_url: str) -> str:
    """Best-effort channel name from a channel URL/handle."""
    parsed = urlparse(channel_url)
    path = parsed.path.strip("/")
    if not path:
        return "YouTube Channel"

    parts = [p for p in path.split("/") if p]
    if not parts:
        return "YouTube Channel"

    if parts[-1] in {
        "videos",
        "shorts",
        "streams",
        "playlists",
        "featured",
        "community",
        "live",
        "releases",
    }:
        parts = parts[:-1]

    if not parts:
        return "YouTube Channel"

    candidate = parts[0]
    if candidate in {"channel", "c", "user"} and len(parts) >= 2:
        candidate = parts[1]

    if candidate.startswith("@"):
        candidate = candidate[1:]

    candidate = candidate.strip()
    return candidate or "YouTube Channel"


def pick_channel_name(data: dict, channel_url: str) -> str:
    """Pick the most reliable channel name available."""
    for key in ("channel", "uploader", "uploader_id", "channel_id"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return infer_channel_name_from_url(channel_url)


def is_short_entry(data: dict) -> bool:
    """Heuristic to identify YouTube Shorts or short-form videos."""
    for key in ("webpage_url", "url", "original_url"):
        value = data.get(key)
        if isinstance(value, str) and "/shorts/" in value:
            return True

    duration = data.get("duration")
    if isinstance(duration, (int, float)) and duration > 0 and duration < MIN_VIDEO_DURATION:
        return True

    title = data.get("title")
    if isinstance(title, str) and "#short" in title.lower():
        return True

    return False


def get_channel_videos(channel_url: str) -> tuple[Optional[ChannelInfo], list[VideoInfo]]:
    """
    Get list of all videos from a YouTube channel.
    
    Returns:
        Tuple of (ChannelInfo, list of VideoInfo)
    """
    logger.info(f"Fetching video list from: {channel_url}")
    
    # Use yt-dlp to get video list with metadata
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        "--no-warnings",
        channel_url
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for large channels
        )
        
        if result.returncode != 0:
            logger.error(f"yt-dlp error: {result.stderr}")
            return None, []
        
        videos = []
        channel_info = None
        
        short_videos_skipped = 0
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                
                if is_short_entry(data):
                    short_videos_skipped += 1
                    continue
                
                # Extract channel info from first video
                if channel_info is None and "channel_id" in data:
                    channel_info = ChannelInfo(
                        id=data.get("channel_id", "unknown"),
                        name=pick_channel_name(data, channel_url),
                        url=channel_url,
                        video_count=0,  # Will update after
                        scraped_at=datetime.now().isoformat()
                    )
                
                video = VideoInfo(
                    id=data.get("id", ""),
                    title=data.get("title", "Unknown"),
                    url=f"https://www.youtube.com/watch?v={data.get('id', '')}",
                    upload_date=data.get("upload_date"),
                    duration=data.get("duration"),
                    description=data.get("description", "")[:500] if data.get("description") else None
                )
                
                if video.id:
                    videos.append(video)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse video entry: {e}")
                continue
        
        if channel_info:
            channel_info.video_count = len(videos)
        
        if short_videos_skipped:
            logger.info(f"Found {len(videos)} videos (skipped {short_videos_skipped} short videos)")
        else:
            logger.info(f"Found {len(videos)} videos")
        return channel_info, videos
        
    except subprocess.TimeoutExpired:
        logger.error("Timeout while fetching video list")
        return None, []
    except Exception as e:
        logger.error(f"Error fetching videos: {e}")
        return None, []


def is_rate_limit_error(stderr_text: str) -> bool:
    """Heuristic check for rate limiting in yt-dlp stderr output."""
    if not stderr_text:
        return False
    text = stderr_text.lower()
    if "429" in text or "too many requests" in text or "rate limit" in text or "temporarily blocked" in text:
        return True
    if "http error 403" in text and ("forbidden" in text or "blocked" in text or "access" in text):
        return True
    return False


def compact_error(text: str, max_len: int = 200) -> str:
    """Trim verbose errors for logs."""
    if not text:
        return "unknown error"
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def find_subtitle_file(temp_dir: Path, video_id: str, lang: str) -> Optional[Path]:
    """Find a subtitle file in a directory for a given video."""
    subtitle_patterns = [
        temp_dir / f"{video_id}.{lang}.json3",
        temp_dir / f"{video_id}.{lang}.vtt",
        temp_dir / f"{video_id}.{lang}.srt",
    ]

    # Also check for auto-generated variants
    for pattern in [f"{video_id}.{lang}*.json3", f"{video_id}*.{lang}.json3"]:
        subtitle_patterns.extend(temp_dir.glob(pattern))

    for pattern in subtitle_patterns:
        if isinstance(pattern, Path) and pattern.exists():
            return pattern

    # Check for any subtitle file for this video
    for candidate in temp_dir.glob(f"{video_id}*"):
        if candidate.suffix in [".json3", ".vtt", ".srt"]:
            return candidate

    return None


def download_transcript(
    video_id: str,
    output_path: Path,
    lang: str = "en"
) -> tuple[bool, Optional[str], Optional[str], bool]:
    """
    Download transcript for a single video.
    
    Args:
        video_id: YouTube video ID
        output_path: Path to save the transcript JSON
        lang: Preferred language code
    
    Returns:
        Tuple of (success, transcript_text, error_message, rate_limited)
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        with tempfile.TemporaryDirectory(dir=output_path.parent, prefix=f".temp_{video_id}_") as tmp_dir:
            temp_dir = Path(tmp_dir)

            # Try to download subtitles (prefer manual, fallback to auto-generated)
            cmd = [
                "yt-dlp",
                "--write-subs",
                "--write-auto-subs",
                "--sub-lang", lang,
                "--sub-format", "json3",
                "--skip-download",
                "--no-warnings",
                "-o", str(temp_dir / "%(id)s.%(ext)s"),
                video_url
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            stderr = (result.stderr or "").strip()
            rate_limited = is_rate_limit_error(stderr)

            if result.returncode != 0:
                if rate_limited:
                    return False, None, f"Rate limited: {compact_error(stderr)}", True
                return False, None, f"yt-dlp error: {compact_error(stderr)}", False

            subtitle_file = find_subtitle_file(temp_dir, video_id, lang)

            if not subtitle_file:
                if rate_limited:
                    return False, None, "Rate limited: no subtitles returned", True
                return False, None, "No subtitles available", False

            # Parse the subtitle file
            transcript_text = parse_subtitle_file(subtitle_file)

            if transcript_text:
                return True, transcript_text, None, False
            return False, None, "Failed to parse subtitles", False

    except subprocess.TimeoutExpired:
        return False, None, "Timeout downloading transcript", False
    except Exception as e:
        return False, None, str(e), False


def run_batch_download(video_urls: list[str], temp_dir: Path, lang: str) -> tuple[bool, bool, Optional[str]]:
    """Run yt-dlp once to download subtitles for a list of video URLs."""
    if not video_urls:
        return True, False, None

    batch_file = temp_dir / "batch_urls.txt"
    batch_file.write_text("\n".join(video_urls), encoding="utf-8")

    cmd = [
        "yt-dlp",
        "--write-subs",
        "--write-auto-subs",
        "--sub-lang", lang,
        "--sub-format", "json3",
        "--skip-download",
        "--no-warnings",
        "-o", str(temp_dir / "%(id)s.%(ext)s"),
        "--batch-file", str(batch_file),
    ]

    timeout = max(300, min(7200, 10 * len(video_urls)))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return False, False, "Timeout downloading batch transcripts"

    stderr = (result.stderr or "").strip()
    rate_limited = is_rate_limit_error(stderr)

    if result.returncode != 0:
        if rate_limited:
            return False, True, f"Rate limited: {compact_error(stderr)}"
        return False, False, f"yt-dlp error: {compact_error(stderr)}"

    return True, rate_limited, None


def parse_subtitle_file(filepath: Path) -> Optional[str]:
    """Parse subtitle file and extract plain text transcript."""
    try:
        content = filepath.read_text(encoding="utf-8")
        
        if filepath.suffix == ".json3":
            return parse_json3_subtitles(content)
        elif filepath.suffix == ".vtt":
            return parse_vtt_subtitles(content)
        elif filepath.suffix == ".srt":
            return parse_srt_subtitles(content)
        else:
            return content
            
    except Exception as e:
        logger.warning(f"Error parsing subtitle file: {e}")
        return None


def parse_json3_subtitles(content: str) -> Optional[str]:
    """Parse YouTube JSON3 subtitle format."""
    try:
        data = json.loads(content)
        
        # Extract text segments
        segments = []
        
        if "events" in data:
            for event in data["events"]:
                if "segs" in event:
                    for seg in event["segs"]:
                        if "utf8" in seg:
                            text = seg["utf8"].strip()
                            if text and text != "\n":
                                segments.append(text)
        
        # Join segments, removing duplicate whitespace
        transcript = " ".join(segments)
        transcript = re.sub(r'\s+', ' ', transcript)
        
        return transcript.strip() if transcript else None
        
    except json.JSONDecodeError:
        return None


def parse_vtt_subtitles(content: str) -> Optional[str]:
    """Parse VTT subtitle format."""
    lines = content.split("\n")
    segments = []
    
    for line in lines:
        line = line.strip()
        # Skip header, timestamps, and empty lines
        if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
            continue
        # Remove VTT tags
        line = re.sub(r'<[^>]+>', '', line)
        if line:
            segments.append(line)
    
    transcript = " ".join(segments)
    transcript = re.sub(r'\s+', ' ', transcript)
    
    return transcript.strip() if transcript else None


def parse_srt_subtitles(content: str) -> Optional[str]:
    """Parse SRT subtitle format."""
    lines = content.split("\n")
    segments = []
    
    for line in lines:
        line = line.strip()
        # Skip numbers, timestamps, and empty lines
        if not line or line.isdigit() or "-->" in line:
            continue
        segments.append(line)
    
    transcript = " ".join(segments)
    transcript = re.sub(r'\s+', ' ', transcript)
    
    return transcript.strip() if transcript else None


def scrape_channel(
    channel_url: str,
    output_dir: Path,
    lang: str = "en",
    resume: bool = True,
    delay: float = 1.0,
    max_workers: int = DEFAULT_MAX_WORKERS,
    mode: str = "per-video"
) -> dict:
    """
    Main function to scrape all transcripts from a channel.
    
    Args:
        channel_url: YouTube channel URL or ID
        output_dir: Directory to save transcripts
        lang: Preferred language code
        resume: Skip already downloaded transcripts
        delay: Delay between requests (seconds)
        mode: "per-video" (parallel) or "batch" (single yt-dlp run)
    
    Returns:
        Summary dict with results
    """
    # Normalize URL
    channel_url = normalize_channel_url(channel_url)
    
    # Ensure yt-dlp is installed
    if not ensure_yt_dlp_installed():
        return {"error": "Failed to install yt-dlp"}
    
    # Get channel videos
    channel_info, videos = get_channel_videos(channel_url)
    
    if not videos:
        return {"error": "No videos found or failed to fetch channel"}
    
    # Create output directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    
    # Load existing index if resuming
    index_path = output_dir / "index.json"
    existing_index = {}
    if resume and index_path.exists():
        try:
            existing_index = json.loads(index_path.read_text())
            logger.info(f"Resuming: found {len(existing_index.get('videos', {}))} previously processed videos")
        except json.JSONDecodeError:
            pass
    
    # Save channel metadata
    if channel_info:
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(asdict(channel_info), indent=2))
        logger.info(f"Saved channel metadata: {channel_info.name}")
    
    # Process each video
    results = {
        "success": 0,
        "no_transcript": 0,
        "error": 0,
        "skipped": 0
    }
    
    processed_videos = {}

    pending_videos = deque()
    processed_count = 0

    for video in videos:
        # Check if already processed
        if resume and video.id in existing_index.get("videos", {}):
            prev_status = existing_index["videos"][video.id].get("transcript_status")
            if prev_status == "success":
                logger.info(f"Skipping (already downloaded): {video.title[:50]}")
                results["skipped"] += 1
                processed_videos[video.id] = existing_index["videos"][video.id]
                processed_count += 1
                if processed_count % 10 == 0:
                    save_index(index_path, channel_info, processed_videos, results)
                continue

        pending_videos.append(video)

    if mode == "batch":
        if delay > 0:
            logger.info("Batch mode ignores --delay (single yt-dlp run per attempt)")

        remaining = list(pending_videos)
        attempt = 0
        rate_limit_hits = 0

        while remaining:
            attempt += 1
            with tempfile.TemporaryDirectory(dir=output_dir, prefix=".batch_subs_") as tmp_dir:
                temp_dir = Path(tmp_dir)
                video_urls = [video.url for video in remaining]
                ok, rate_limited, batch_error = run_batch_download(video_urls, temp_dir, lang)

                if batch_error:
                    logger.warning(f"Batch download issue: {batch_error}")

                retry_list = []
                for video in remaining:
                    subtitle_file = find_subtitle_file(temp_dir, video.id, lang)
                    transcript_text = parse_subtitle_file(subtitle_file) if subtitle_file else None

                    if transcript_text:
                        video.transcript_status = "success"
                        transcript_data = {
                            "video_id": video.id,
                            "title": video.title,
                            "url": video.url,
                            "upload_date": video.upload_date,
                            "duration": video.duration,
                            "transcript": transcript_text,
                            "word_count": len(transcript_text.split()),
                            "scraped_at": datetime.now().isoformat()
                        }
                        transcript_path = transcripts_dir / f"{video.id}.json"
                        transcript_path.write_text(json.dumps(transcript_data, indent=2, ensure_ascii=False))
                        results["success"] += 1
                        logger.info(f"  ✓ Saved transcript ({len(transcript_text.split())} words)")
                    elif rate_limited and attempt <= MAX_RETRIES:
                        retry_list.append(video)
                        continue
                    else:
                        if not ok and batch_error:
                            video.transcript_status = "error"
                            video.error_message = batch_error
                            results["error"] += 1
                            logger.warning(f"  ✗ Error: {batch_error}")
                        elif rate_limited:
                            video.transcript_status = "error"
                            video.error_message = "Rate limited (batch)"
                            results["error"] += 1
                            logger.warning("  ✗ Error: Rate limited (batch)")
                        else:
                            video.transcript_status = "no_transcript"
                            video.error_message = "No subtitles available"
                            results["no_transcript"] += 1
                            logger.info("  ⊘ No transcript available")

                    processed_videos[video.id] = asdict(video)
                    processed_count += 1
                    if processed_count % 10 == 0:
                        save_index(index_path, channel_info, processed_videos, results)

            if retry_list and rate_limited and attempt <= MAX_RETRIES:
                rate_limit_hits += 1
                backoff = min(
                    RATE_LIMIT_BACKOFF_MAX,
                    RATE_LIMIT_BACKOFF_BASE * (2 ** min(rate_limit_hits - 1, 4))
                )
                logger.warning(
                    f"Rate limited; backing off {backoff:.0f}s and retrying batch "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                time.sleep(backoff)
                remaining = retry_list
                continue

            remaining = retry_list

    else:
        attempts = {video.id: 0 for video in pending_videos}
        max_workers = max(1, int(max_workers))
        logger.info(f"Starting downloads with up to {max_workers} concurrent workers")

        in_flight = {}
        started_count = 0
        rate_limit_hits = 0
        cooldown_until = 0.0
        last_submit_time = 0.0
        concurrency = max_workers

        def worker(video_info: VideoInfo) -> dict:
            transcript_path = transcripts_dir / f"{video_info.id}.json"
            success, transcript, error, rate_limited = download_transcript(video_info.id, transcript_path, lang)
            return {
                "video": video_info,
                "success": success,
                "transcript": transcript,
                "error": error,
                "rate_limited": rate_limited
            }

        def submit_video(executor: ThreadPoolExecutor, video_info: VideoInfo):
            nonlocal started_count, last_submit_time
            # Global cooldown if rate limited
            if time.time() < cooldown_until:
                time.sleep(max(0.0, cooldown_until - time.time()))

            # Throttle submission cadence
            if delay > 0:
                since_last = time.time() - last_submit_time if last_submit_time else None
                if since_last is not None and since_last < delay:
                    time.sleep(delay - since_last)
            last_submit_time = time.time()

            started_count += 1
            logger.info(f"[{started_count}/{len(videos)}] Processing: {video_info.title[:50]}...")
            future = executor.submit(worker, video_info)
            in_flight[future] = video_info

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while pending_videos or in_flight:
                while pending_videos and len(in_flight) < concurrency:
                    submit_video(executor, pending_videos.popleft())

                if not in_flight:
                    continue

                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    video = in_flight.pop(future)
                    try:
                        result = future.result()
                    except Exception as e:
                        video.transcript_status = "error"
                        video.error_message = f"Worker error: {e}"
                        results["error"] += 1
                        processed_videos[video.id] = asdict(video)
                        processed_count += 1
                        continue

                    if result["rate_limited"]:
                        attempts[video.id] += 1
                        if attempts[video.id] <= MAX_RETRIES:
                            rate_limit_hits += 1
                            concurrency = max(1, concurrency - 1)
                            backoff = min(
                                RATE_LIMIT_BACKOFF_MAX,
                                RATE_LIMIT_BACKOFF_BASE * (2 ** min(rate_limit_hits - 1, 4))
                            )
                            cooldown_until = max(cooldown_until, time.time() + backoff)
                            logger.warning(
                                f"Rate limited; backing off {backoff:.0f}s and reducing workers to {concurrency} "
                                f"(retry {attempts[video.id]}/{MAX_RETRIES})"
                            )
                            pending_videos.append(video)
                            continue
                        else:
                            result["error"] = result["error"] or "Rate limited (max retries exceeded)"

                    if result["success"] and result["transcript"]:
                        video.transcript_status = "success"
                        transcript_text = result["transcript"]

                        # Save transcript with metadata
                        transcript_data = {
                            "video_id": video.id,
                            "title": video.title,
                            "url": video.url,
                            "upload_date": video.upload_date,
                            "duration": video.duration,
                            "transcript": transcript_text,
                            "word_count": len(transcript_text.split()),
                            "scraped_at": datetime.now().isoformat()
                        }
                        transcript_path = transcripts_dir / f"{video.id}.json"
                        transcript_path.write_text(json.dumps(transcript_data, indent=2, ensure_ascii=False))

                        results["success"] += 1
                        logger.info(f"  ✓ Saved transcript ({len(transcript_text.split())} words)")

                    elif result["error"] and "No subtitles" in result["error"]:
                        video.transcript_status = "no_transcript"
                        video.error_message = result["error"]
                        results["no_transcript"] += 1
                        logger.info(f"  ⊘ No transcript available")

                    else:
                        video.transcript_status = "error"
                        video.error_message = result["error"]
                        results["error"] += 1
                        logger.warning(f"  ✗ Error: {result['error']}")

                    processed_videos[video.id] = asdict(video)
                    processed_count += 1

                    # Save index periodically (every 10 processed videos)
                    if processed_count % 10 == 0:
                        save_index(index_path, channel_info, processed_videos, results)
    
    # Final save
    save_index(index_path, channel_info, processed_videos, results)

    # Generate combined LLM-ready transcript
    combined_path = generate_combined_transcript(output_dir, channel_info)

    # Summary
    logger.info("=" * 50)
    logger.info("Scraping complete!")
    logger.info(f"  Total videos: {len(videos)}")
    logger.info(f"  Successful: {results['success']}")
    logger.info(f"  No transcript: {results['no_transcript']}")
    logger.info(f"  Errors: {results['error']}")
    logger.info(f"  Skipped (resumed): {results['skipped']}")
    logger.info(f"  Output directory: {output_dir}")
    if combined_path:
        logger.info(f"  LLM-ready file: {combined_path}")

    return {
        "channel": asdict(channel_info) if channel_info else None,
        "total_videos": len(videos),
        "results": results,
        "output_dir": str(output_dir),
        "combined_transcript": str(combined_path) if combined_path else None
    }


def save_index(index_path: Path, channel_info: Optional[ChannelInfo], videos: dict, results: dict):
    """Save the index file with current progress."""
    index_data = {
        "channel": asdict(channel_info) if channel_info else None,
        "last_updated": datetime.now().isoformat(),
        "results": results,
        "videos": videos
    }
    index_path.write_text(json.dumps(index_data, indent=2, ensure_ascii=False))


def generate_combined_transcript(output_dir: Path, channel_info: Optional[ChannelInfo]) -> Optional[Path]:
    """
    Generate a single combined text file with all transcripts, ready for LLM input.

    Args:
        output_dir: Directory containing the transcripts folder
        channel_info: Channel metadata

    Returns:
        Path to the combined transcript file, or None if no transcripts found
    """
    transcripts_dir = output_dir / "transcripts"
    if not transcripts_dir.exists():
        return None

    # Collect all transcripts with metadata
    transcript_entries = []

    for json_file in sorted(transcripts_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            if data.get("transcript"):
                transcript_entries.append({
                    "title": data.get("title", "Unknown"),
                    "url": data.get("url", ""),
                    "upload_date": data.get("upload_date"),
                    "duration": data.get("duration"),
                    "word_count": data.get("word_count", 0),
                    "transcript": data["transcript"]
                })
        except (json.JSONDecodeError, KeyError):
            continue

    if not transcript_entries:
        return None

    # Sort by upload date if available, otherwise by title
    transcript_entries.sort(
        key=lambda x: x.get("upload_date") or "0000",
        reverse=True
    )

    # Build the combined text file
    lines = []

    # Header
    if channel_info:
        channel_name = channel_info.name or infer_channel_name_from_url(channel_info.url)
    else:
        channel_name = "YouTube Channel"
    lines.append(f"# {channel_name} - Complete Transcripts")
    lines.append(f"")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Total videos: {len(transcript_entries)}")
    total_words = sum(t["word_count"] for t in transcript_entries)
    lines.append(f"Total words: {total_words:,}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    # Each transcript
    for i, entry in enumerate(transcript_entries, 1):
        lines.append(f"## [{i}/{len(transcript_entries)}] {entry['title']}")
        lines.append(f"URL: {entry['url']}")
        if entry.get("upload_date"):
            # Format date nicely if possible
            try:
                date_str = entry["upload_date"]
                if len(date_str) == 8:  # YYYYMMDD format
                    formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                    lines.append(f"Date: {formatted}")
            except:
                lines.append(f"Date: {entry['upload_date']}")
        if entry.get("duration") is not None:
            try:
                duration = int(entry["duration"])
                mins = duration // 60
                secs = duration % 60
                lines.append(f"Duration: {mins}:{secs:02d}")
            except (TypeError, ValueError):
                pass
        lines.append(f"Words: {entry['word_count']:,}")
        lines.append("")
        lines.append(entry["transcript"])
        lines.append("")
        lines.append("-" * 80)
        lines.append("")

    # Write combined file
    combined_path = output_dir / "all_transcripts.txt"
    combined_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"Generated combined transcript: {combined_path}")
    logger.info(f"  {len(transcript_entries)} videos, {total_words:,} total words")

    return combined_path


def main():
    parser = argparse.ArgumentParser(
        description="Scrape all video transcripts from a YouTube channel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s https://www.youtube.com/@channelname
    %(prog)s UCxxxxxxxxxx --output-dir ./my_transcripts
    %(prog)s @channelhandle --lang es --resume
    %(prog)s @channelhandle --mode batch
        """
    )
    
    parser.add_argument(
        "channel",
        help="YouTube channel URL, ID, or @handle"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="./transcripts",
        help="Output directory (default: ./transcripts)"
    )
    
    parser.add_argument(
        "--lang", "-l",
        default="en",
        help="Preferred language code (default: en)"
    )
    
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        default=True,
        help="Resume from previous run (skip already downloaded)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume, re-download everything"
    )
    
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Max concurrent workers (default: {DEFAULT_MAX_WORKERS})"
    )

    parser.add_argument(
        "--mode",
        choices=["per-video", "batch"],
        default="per-video",
        help="Download mode: per-video (parallel) or batch (single yt-dlp run)"
    )
    
    args = parser.parse_args()
    
    resume = not args.no_resume
    
    result = scrape_channel(
        channel_url=args.channel,
        output_dir=Path(args.output_dir),
        lang=args.lang,
        resume=resume,
        delay=args.delay,
        max_workers=args.max_workers,
        mode=args.mode
    )
    
    if "error" in result:
        logger.error(f"Failed: {result['error']}")
        sys.exit(1)
    
    # Print summary as JSON for programmatic use
    print("\n--- SUMMARY (JSON) ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

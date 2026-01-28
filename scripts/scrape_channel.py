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
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


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


def normalize_channel_url(channel_input: str) -> str:
    """Convert various channel input formats to a consistent URL."""
    # Already a full URL
    if channel_input.startswith("http"):
        return channel_input
    
    # Channel ID (starts with UC)
    if channel_input.startswith("UC") and len(channel_input) == 24:
        return f"https://www.youtube.com/channel/{channel_input}"
    
    # Handle @username format
    if channel_input.startswith("@"):
        return f"https://www.youtube.com/{channel_input}"
    
    # Assume it's a channel name/handle
    return f"https://www.youtube.com/@{channel_input}"


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
        
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                
                # Extract channel info from first video
                if channel_info is None and "channel_id" in data:
                    channel_info = ChannelInfo(
                        id=data.get("channel_id", "unknown"),
                        name=data.get("channel", data.get("uploader", "Unknown Channel")),
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
        
        logger.info(f"Found {len(videos)} videos")
        return channel_info, videos
        
    except subprocess.TimeoutExpired:
        logger.error("Timeout while fetching video list")
        return None, []
    except Exception as e:
        logger.error(f"Error fetching videos: {e}")
        return None, []


def download_transcript(video_id: str, output_path: Path, lang: str = "en") -> tuple[bool, Optional[str], Optional[str]]:
    """
    Download transcript for a single video.
    
    Args:
        video_id: YouTube video ID
        output_path: Path to save the transcript JSON
        lang: Preferred language code
    
    Returns:
        Tuple of (success, transcript_text, error_message)
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Create temp directory for subtitle files
    temp_dir = output_path.parent / ".temp"
    temp_dir.mkdir(exist_ok=True)
    
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
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Look for downloaded subtitle file
        subtitle_patterns = [
            temp_dir / f"{video_id}.{lang}.json3",
            temp_dir / f"{video_id}.{lang}.vtt",
            temp_dir / f"{video_id}.{lang}.srt",
        ]
        
        # Also check for auto-generated variants
        for pattern in [f"{video_id}.{lang}*.json3", f"{video_id}*.{lang}.json3"]:
            subtitle_patterns.extend(temp_dir.glob(pattern))
        
        subtitle_file = None
        for pattern in subtitle_patterns:
            if isinstance(pattern, Path) and pattern.exists():
                subtitle_file = pattern
                break
        
        if not subtitle_file:
            # Check for any subtitle file for this video
            for f in temp_dir.glob(f"{video_id}*"):
                if f.suffix in [".json3", ".vtt", ".srt"]:
                    subtitle_file = f
                    break
        
        if not subtitle_file:
            return False, None, "No subtitles available"
        
        # Parse the subtitle file
        transcript_text = parse_subtitle_file(subtitle_file)
        
        # Clean up temp file
        subtitle_file.unlink()
        
        if transcript_text:
            return True, transcript_text, None
        else:
            return False, None, "Failed to parse subtitles"
            
    except subprocess.TimeoutExpired:
        return False, None, "Timeout downloading transcript"
    except Exception as e:
        return False, None, str(e)


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
    delay: float = 1.0
) -> dict:
    """
    Main function to scrape all transcripts from a channel.
    
    Args:
        channel_url: YouTube channel URL or ID
        output_dir: Directory to save transcripts
        lang: Preferred language code
        resume: Skip already downloaded transcripts
        delay: Delay between requests (seconds)
    
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
    
    for i, video in enumerate(videos, 1):
        logger.info(f"[{i}/{len(videos)}] Processing: {video.title[:50]}...")
        
        # Check if already processed
        if resume and video.id in existing_index.get("videos", {}):
            prev_status = existing_index["videos"][video.id].get("transcript_status")
            if prev_status == "success":
                logger.info(f"  Skipping (already downloaded)")
                results["skipped"] += 1
                processed_videos[video.id] = existing_index["videos"][video.id]
                continue
        
        # Download transcript
        transcript_path = transcripts_dir / f"{video.id}.json"
        success, transcript, error = download_transcript(video.id, transcript_path, lang)
        
        if success and transcript:
            video.transcript_status = "success"
            
            # Save transcript with metadata
            transcript_data = {
                "video_id": video.id,
                "title": video.title,
                "url": video.url,
                "upload_date": video.upload_date,
                "duration": video.duration,
                "transcript": transcript,
                "word_count": len(transcript.split()),
                "scraped_at": datetime.now().isoformat()
            }
            transcript_path.write_text(json.dumps(transcript_data, indent=2, ensure_ascii=False))
            
            results["success"] += 1
            logger.info(f"  ✓ Saved transcript ({len(transcript.split())} words)")
            
        elif error and "No subtitles" in error:
            video.transcript_status = "no_transcript"
            video.error_message = error
            results["no_transcript"] += 1
            logger.info(f"  ⊘ No transcript available")
            
        else:
            video.transcript_status = "error"
            video.error_message = error
            results["error"] += 1
            logger.warning(f"  ✗ Error: {error}")
        
        processed_videos[video.id] = asdict(video)
        
        # Save index periodically (every 10 videos)
        if i % 10 == 0:
            save_index(index_path, channel_info, processed_videos, results)
        
        # Rate limiting
        time.sleep(delay)
    
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
    channel_name = channel_info.name if channel_info else "YouTube Channel"
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
        if entry.get("duration"):
            mins = entry["duration"] // 60
            secs = entry["duration"] % 60
            lines.append(f"Duration: {mins}:{secs:02d}")
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
    
    args = parser.parse_args()
    
    resume = not args.no_resume
    
    result = scrape_channel(
        channel_url=args.channel,
        output_dir=Path(args.output_dir),
        lang=args.lang,
        resume=resume,
        delay=args.delay
    )
    
    if "error" in result:
        logger.error(f"Failed: {result['error']}")
        sys.exit(1)
    
    # Print summary as JSON for programmatic use
    print("\n--- SUMMARY (JSON) ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

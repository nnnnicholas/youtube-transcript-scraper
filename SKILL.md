---
name: youtube-transcript-scraper
description: Scrape transcripts from all videos on a YouTube channel. Use when the user wants to download, extract, or scrape captions/subtitles/transcripts from YouTube channels, playlists, or multiple videos. Handles both manual and auto-generated captions with resume support for large channels.
---

# YouTube Transcript Scraper

Scrapes all video transcripts from a YouTube channel using `yt-dlp`.

## Quick Start

```bash
python scripts/scrape_channel.py <channel_url_or_id> --output-dir ./output
```

**Accepted inputs:**
- Full URL: `https://www.youtube.com/@channelname`
- Channel ID: `UCxxxxxxxxxxxxxxxxxx`
- Handle: `@channelname` or just `channelname`

## Features

- Downloads manual captions when available, falls back to auto-generated
- Resumes from previous runs (skips already downloaded)
- Rate limiting to avoid being blocked
- Parallel downloads (default 3 workers) with backoff on rate limits
- Skips YouTube Shorts and short videos (<4 minutes)
- Optional batch mode (single yt-dlp run) for faster downloads
- Structured JSON output with metadata
- Progress logging

## Output Structure

```
output_dir/
├── metadata.json       # Channel info
├── index.json          # Progress tracker with all video statuses  
└── transcripts/
    ├── {video_id}.json # Each transcript with metadata
    └── ...
```

Each transcript file contains:
```json
{
  "video_id": "abc123",
  "title": "Video Title",
  "url": "https://youtube.com/watch?v=abc123",
  "upload_date": "20240115",
  "duration": 600,
  "transcript": "Full transcript text...",
  "word_count": 1234,
  "scraped_at": "2024-01-15T10:30:00"
}
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--output-dir`, `-o` | Output directory | `./transcripts` |
| `--lang`, `-l` | Preferred language code | `en` |
| `--delay`, `-d` | Seconds between requests | `1.0` |
| `--max-workers` | Max concurrent workers | `3` |
| `--mode` | Download mode: `per-video` or `batch` | `per-video` |
| `--no-resume` | Re-download everything | (resume enabled) |

**Note:** Shorts and short videos are skipped by default (detected by `/shorts/` URLs, `#shorts` titles, or duration < 4 minutes).

## Examples

```bash
# Basic usage
python scripts/scrape_channel.py https://www.youtube.com/@TechChannel

# Spanish transcripts
python scripts/scrape_channel.py @SpanishPodcast --lang es -o ./spanish_transcripts

# Faster scraping (reduce delay) - use with caution
python scripts/scrape_channel.py UCxxxxx --delay 0.5

# Batch mode (single yt-dlp run)
python scripts/scrape_channel.py @ChannelHandle --mode batch
```

## Programmatic Usage

```python
from pathlib import Path
from scripts.scrape_channel import scrape_channel

result = scrape_channel(
    channel_url="https://www.youtube.com/@channelname",
    output_dir=Path("./output"),
    lang="en",
    resume=True,
    delay=1.0,
    max_workers=3,
    mode="per-video"
)

print(f"Downloaded {result['results']['success']} transcripts")
```

## Troubleshooting

See [references/troubleshooting.md](references/troubleshooting.md) for common issues.

**Common issues:**
- `No videos found`: Check URL format, channel may be private
- `No subtitles available`: Video has no captions (manual or auto)
- Timeout errors: Increase `--delay`, try again later

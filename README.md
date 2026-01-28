# YouTube Transcript Scraper

Scrape all video transcripts from a YouTube channel into a single text file you can paste directly into an LLM.

## What it does

1. Fetches all videos from a YouTube channel
2. Downloads transcripts (manual captions preferred, auto-generated as fallback)
3. Saves individual JSON files with metadata
4. **Generates `all_transcripts.txt`** - a combined file ready to copy-paste into ChatGPT, Claude, etc.

## Quick Start

```bash
# Clone and run
git clone https://github.com/nnnnicholas/youtube-transcript-scraper.git
cd youtube-transcript-scraper

python scripts/scrape_channel.py https://www.youtube.com/@channelname
```

The script auto-installs `yt-dlp` if needed.

## Output

```
output_dir/
├── all_transcripts.txt   # <-- LLM-ready! Just copy-paste this
├── metadata.json         # Channel info
├── index.json            # Progress tracker
└── transcripts/
    └── {video_id}.json   # Individual transcripts with metadata
```

### all_transcripts.txt format

```
# Channel Name - Complete Transcripts

Generated: 2025-01-28 12:00
Total videos: 42
Total words: 150,000

================================================================================

## [1/42] Video Title Here
URL: https://youtube.com/watch?v=abc123
Date: 2025-01-15
Duration: 12:34
Words: 3,456

The full transcript text goes here...

--------------------------------------------------------------------------------

## [2/42] Another Video
...
```

## Usage

```bash
# Basic - scrapes to ./transcripts
python scripts/scrape_channel.py @channelname

# Custom output directory
python scripts/scrape_channel.py https://www.youtube.com/@channelname -o ./my_output

# Spanish transcripts
python scripts/scrape_channel.py @channelname --lang es

# Re-download everything (ignore previous progress)
python scripts/scrape_channel.py @channelname --no-resume

# Faster (but may get rate-limited)
python scripts/scrape_channel.py @channelname --delay 0.5

# Batch mode (single yt-dlp run)
python scripts/scrape_channel.py @channelname --mode batch
```

### Input formats

All of these work:
- `https://www.youtube.com/@channelname`
- `https://www.youtube.com/channel/UCxxxxxxxxxx`
- `@channelname`
- `channelname`
- `UCxxxxxxxxxxxxxxxxxx` (channel ID)

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--output-dir`, `-o` | Output directory | `./transcripts` |
| `--lang`, `-l` | Preferred language code | `en` |
| `--delay`, `-d` | Seconds between requests | `1.0` |
| `--max-workers` | Max concurrent workers | `3` |
| `--mode` | Download mode: `per-video` or `batch` | `per-video` |
| `--no-resume` | Re-download everything | (resume enabled) |

**Note:** Shorts and short videos are skipped by default (detected by `/shorts/` URLs, `#shorts` titles, or duration < 4 minutes).

## Features

- **Resume support**: Interrupted? Just run again - skips already downloaded videos
- **Auto-generated captions**: Falls back to auto-captions when manual aren't available
- **Progress saving**: Saves every 10 videos in case of interruption
- **Parallel downloads**: Configurable workers (default 3) with backoff and retry on rate limits
- **Batch mode**: Single yt-dlp run for faster downloads on some channels
- **Skips shorts**: Shorts and videos under 4 minutes are excluded by default

## Use cases

- Feed an entire channel's content to an LLM for analysis
- Create a knowledge base from educational content
- Research what topics a channel covers
- Generate summaries or find specific information across videos

## Requirements

- Python 3.8+
- `yt-dlp` (auto-installed if missing)

## Troubleshooting

See [references/troubleshooting.md](references/troubleshooting.md) for common issues.

**Common problems:**
- `No videos found`: Check URL format, channel may be private
- `No subtitles available`: Video has no captions (manual or auto)
- Timeout errors: Increase `--delay`, try again later

## License

MIT

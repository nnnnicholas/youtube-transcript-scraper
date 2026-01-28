# Troubleshooting Guide

## Common Issues

### "No videos found"

**Causes:**
- Channel URL is malformed
- Channel is private or deleted
- YouTube rate limiting

**Solutions:**
1. Verify URL opens in browser
2. Try channel ID format: `UCxxxxxxxxxx`
3. Wait and retry if rate limited

### "No subtitles available"

Some videos simply don't have captions. Check `index.json` for `no_transcript` entries.

**Possible reasons:**
- Video is very new (auto-captions not yet generated)
- Creator disabled captions
- Audio is music-only or non-speech

### Timeout errors

**Solutions:**
1. Increase delay: `--delay 2.0`
2. Check internet connection
3. Run with `--resume` to continue from where it stopped

### Rate limiting (403 errors)

YouTube may block requests if too fast.

**Solutions:**
1. Increase delay: `--delay 3.0`
2. Wait 1-2 hours and resume
3. Use a VPN (different IP)

### Wrong language transcripts

Auto-generated captions may not match `--lang` if:
- Original audio is different language
- Manual captions only exist in one language

**Solution:** Check what languages are available on a sample video, then use that `--lang` code.

## Performance Tips

### Large channels (1000+ videos)

- Use `--delay 1.5` or higher
- Run during off-peak hours
- Use `--resume` if interrupted
- Check `index.json` for progress

### Disk space

Estimate: ~5-10KB per transcript on average. A 1000-video channel needs ~10MB.

## Verifying Results

Check `index.json` for summary:
```bash
cat output/index.json | python -m json.tool | head -20
```

Count successful downloads:
```bash
ls output/transcripts/*.json | wc -l
```

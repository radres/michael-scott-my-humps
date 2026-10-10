[![Michael Scott's "My Humps" ringtone from The Office](https://img.youtube.com/vi/w6szQQEFzC8/hqdefault.jpg)](https://www.youtube.com/watch?v=w6szQQEFzC8)

# My Humps: the Michael Scott ringtone, extended

Michael's ringtone in The Office (season 3, "The Convention") only exists as a ~23 second clip.
This is a 3 minute version made from the ringtone recording itself, cut at the bar lines and played in a longer order.
Every sound in it is the original ringtone.

**[▶ Listen](https://radres.github.io/michael-scott-my-humps/)** · [mp3](my-humps-extended.mp3)

Show footage of Michael with the song: [the office || my humps](https://www.youtube.com/watch?v=XBdxS1wFFdQ)

## Structure

The ringtone is 12 bars at 124 BPM: a 4 bar hook (bars 0–3), then the chorus twice (bars 4–7 and 8–11). Bar 11 is cut off in every copy online.

| time | bars |
| --- | --- |
| 0:00 | the ringtone as recorded, bars 0–10, then bar 7 instead of the cut-off bar 11 |
| 0:23 | hook stretched to 8 bars (0 1 2 1 2 1 2 3) |
| 0:39 | chorus ×2 |
| 0:54 | hook |
| 1:02 | chorus ×2 |
| 1:17 | same 28 bars again |
| 2:12 | same 28 bars again, fading out over the last 2 bars |

## How it was made

1. `yt-dlp` pulls the ringtone mp3 from the SoundCloud upload linked under the YouTube video. It is the same recording one generation cleaner than YouTube's audio.
2. The bar length comes from cross-correlating bar 4 with its repeat in bar 8: 1.93499 s, 124.03 BPM.
3. `extend.py` cuts 45 ms before each downbeat with 25 ms crossfades, so no attack is clipped. The first 21 s is bit-identical to the source, the splices have no clicks, and the beat timing matches the original.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python extend.py
```

There is also a [synth remix](my-humps-synth-remix.mp3) ([MIDI](my-humps-synth-remix.mid)): a full re-arrangement of the transcribed melody, synthesized from scratch by `synth_remix.py`.

Fan edit, non-commercial. The ringtone is a rendition of "My Humps" by the Black Eyed Peas, and The Office belongs to NBC.

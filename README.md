[![Michael Scott's "My Humps" ringtone from The Office](https://img.youtube.com/vi/w6szQQEFzC8/hqdefault.jpg)](https://www.youtube.com/watch?v=w6szQQEFzC8)

# My Humps: the Michael Scott ringtone, extended

Michael's ringtone in The Office (season 3, "The Convention") only exists as a ~23 second clip.
This is a 3 minute version built from that melody.

**[▶ Listen](https://radres.github.io/michael-scott-my-humps/)** · [mp3](my-humps-extended.mp3) · [MIDI](my-humps-extended.mid)

Show footage of Michael with the song: [the office || my humps](https://www.youtube.com/watch?v=XBdxS1wFFdQ)

## Structure

124 BPM, A minor, last third a whole step up in B minor.

| time | section |
| --- | --- |
| 0:00 | intro: the chorus on music box, filter opening |
| 0:15 | hook (the part you hear in the show) |
| 0:23 | chorus ×2 |
| 0:39 | verse: new melody over Am F G E |
| 0:54 | hook + chorus ×2, octave bells |
| 1:17 | solo: the A B C motif climbs to C D E, then E F G |
| 1:33 | chorus ×2 with harmony and arpeggios |
| 1:48 | breakdown: hook at half speed, riser |
| 2:04 | key change: chorus ×2, hook, chorus ×2 |
| 2:43 | outro |

## How it was made

1. Pulled the ringtone audio from the YouTube clip and transcribed it with [basic-pitch](https://github.com/spotify/basic-pitch) plus CQT spectrograms on a 124 BPM grid (`tools/`).
2. The ringtone melody: `_ A B C | B . A~ (G A E) | A A A A F E F E | . D D E F E F E`, over an A pedal with the bass doubling A B C B A.
3. `build.py` writes the arrangement and synthesizes every sound with numpy: PolyBLEP saw lead, FM bell, sine and saw bass, drums from noise and pitch sweeps, convolution reverb, ping-pong delay and kick sidechain. ffmpeg normalizes it to -14 LUFS.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python build.py
```

Fan-made and non-commercial. "My Humps" is by the Black Eyed Peas and The Office belongs to NBC. The repo contains no original audio, only a new synthesized arrangement.

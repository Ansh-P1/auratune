# AuraTune Validation Traces

## Quiet room + podcast
- Detected: noise=`quiet`, content=`podcast`, ambient=-54.0 dB
- Deltas: {'volume_db': 0.0, 'bass_gain_db': 0.0, 'presence_gain_db': 0.0, 'treble_gain_db': 0.0}
- Explanation: "No change needed -- your podcast curve already fits a quiet room."
- Curve plot: `quiet_podcast.png`

## Noisy environment + music
- Detected: noise=`noisy`, content=`music`, ambient=-21.9 dB
- Deltas: {'volume_db': 0.0, 'bass_gain_db': -2.5, 'presence_gain_db': 3.5, 'treble_gain_db': 0.0}
- Explanation: "Because of a noisy environment during music, I boosted vocal clarity by 3.5 dB, pulled bass back 2.5 dB."
- Curve plot: `noisy_music.png`

## Home + movie
- Detected: noise=`moderate`, content=`movie`, ambient=-36.5 dB
- Deltas: {'volume_db': 0.0, 'bass_gain_db': -1.0, 'presence_gain_db': 1.5, 'treble_gain_db': 0.0}
- Explanation: "Because of a moderate environment during movie, I boosted vocal clarity by 1.5 dB, pulled bass back 1.0 dB."
- Curve plot: `home_movie.png`

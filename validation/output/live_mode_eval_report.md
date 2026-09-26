# Live Mode evaluation report

Scripted sequence: `quiet_podcast -> quiet_podcast -> quiet_podcast -> quiet_podcast -> noisy_music -> noisy_music -> noisy_music -> noisy_music -> quiet_podcast -> quiet_podcast -> quiet_podcast -> quiet_podcast` (12 ticks, 5.0s/tick simulated)

- Expected confirmations (startup + real transitions): 3 (at ticks [0, 4, 8])
- Confirmed changes: 3 (at ticks [1, 5, 9])
- **Reaction latency**: avg 1.0 ticks (5.0s) after the room actually changed
- **Unnecessary flip-flops**: 0

## Transition smoothness

- tick 1: avg step 0.00 dB/step, max step 0.00 dB, final step exact: yes
- tick 5: avg step 0.60 dB/step, max step 0.60 dB, final step exact: yes
- tick 9: avg step 0.60 dB/step, max step 0.60 dB, final step exact: yes
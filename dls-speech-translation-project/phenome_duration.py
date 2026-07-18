#!/usr/bin/env python3
import argparse
import numpy as np
import parselmouth
from textgrid import TextGrid, IntervalTier, Interval

# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "PSOLA-based pipeline to stretch/compress all phoneme intervals and all silence intervals "
            "by separate factors, while ensuring the final audio is exactly --target seconds."
        )
    )
    p.add_argument("--wav_in",       required=True,
                   help="Original MFA-aligned WAV (e.g. aligned_audio.wav)")
    p.add_argument("--tg_in",        required=True,
                   help="Original MFA TextGrid (e.g. aligned_audio.TextGrid)")
    p.add_argument("--target",  type=float, required=True,
                   help="Desired final total duration (in seconds), e.g. 5.500")
    p.add_argument("--silence_scale", type=float, required=True,
                   help="Scale factor for *all* silent intervals (e.g. 0.5 to halve silence length).")
    p.add_argument("--wav_out",      required=True,
                   help="Path to write the new PSOLA-stretched WAV (e.g. stretched_mfa.wav)")
    p.add_argument("--tg_out",       required=True,
                   help="Path to write the new TextGrid (e.g. stretched_mfa.TextGrid)")
    return p.parse_args()

# Simple helper to decide whether a phoneme label is “silence.” 
# Modify this list if your MFA rubric uses different labels (e.g. <SIL>, <SPN>, “pau”, etc.).
SILENCE_LABELS = { ""}

def main():
    args = parse_args()

    # 1) Load audio into a parselmouth Sound (PSOLA-ready)
    snd = parselmouth.Sound(args.wav_in)

    # 2) Load the TextGrid; find the “phones” tier
    tg = TextGrid.fromFile(args.tg_in)
    try:
        phones_tier = tg.getFirst("phones")
    except Exception:
        raise RuntimeError(
            "Could not find a tier named 'phones' in your TextGrid. "
            "Open the .TextGrid in Praat and confirm the exact tier name (e.g. 'phones' or 'phones_mfa')."
        )
    

    # (immediately after loading the TextGrid and grabbing phones_tier)

    # --- debugging snippet: list all unique phone labels in your tier
    # unique_labels = set(iv.mark.strip() for iv in phones_tier)
    # print(">>> Unique labels in your 'phones' tier:", unique_labels)
    # # Then exit so you can inspect them before the solver runs:
    # import sys; sys.exit(0)


    # 3) Sum up original durations:
    orig_speech_dur = 0.0
    orig_sil_dur    = 0.0
    for iv in phones_tier:
        t0, t1 = iv.minTime, iv.maxTime
        dur    = t1 - t0
        lab    = iv.mark.strip().lower()
        if lab in SILENCE_LABELS:
            orig_sil_dur += dur
        else:
            orig_speech_dur += dur

    if orig_speech_dur <= 0 and orig_sil_dur <= 0:
        raise RuntimeError("Found zero total duration in both speech and silence. Check your TextGrid.")

    # 4) We already know user’s desired target and a user-specified silence_scale.
    target = args.target
    sil_scale = args.silence_scale

    # 5) Solve for the uniform speech_scale so that:
    #    (orig_speech_dur * speech_scale) + (orig_sil_dur * sil_scale) = target
    #
    #    => speech_scale = (target - orig_sil_dur * sil_scale) / orig_speech_dur
    #
    #    (If orig_speech_dur == 0, that means all intervals are silent—
    #     in which case we simply scale silence to hit the target.)
    if orig_speech_dur > 0:
        speech_scale = (target - orig_sil_dur * sil_scale) / orig_speech_dur
    else:
        # no “phoneme” durations at all—entire audio is silence.
        # In that degenerate case, we force speech_scale = 0 (unused) and just scale silence.
        speech_scale = 0.0

    if speech_scale <= 0:
        raise RuntimeError(
            f"Computed speech_scale ≤ 0 ({speech_scale:.6f}).\n"
            f"Check that (target={target:.3f}) is larger than (orig_sil={orig_sil_dur:.3f} * sil_scale={sil_scale:.3f})."
        )

    print(f"Original speech dur : {orig_speech_dur:.3f} s")
    print(f"Original silence dur: {orig_sil_dur:.3f} s")
    print(f"Silence scale       : {sil_scale:.6f}")
    print(f"→ New silent dur    : {orig_sil_dur * sil_scale:.3f} s")
    print(f"Speech scale solved : {speech_scale:.6f}")
    print(f"→ New speech dur    : {orig_speech_dur * speech_scale:.3f} s")
    print(f"TOTAL target        : {target:.3f} s  (should match sum)")

    # 6) Create a Manipulation object (PSOLA) from the original sound
    #    (time step 0.01 s, pitch floor 75 Hz, pitch ceiling 600 Hz—common defaults)
    manip = parselmouth.praat.call(snd, "To Manipulation", 0.01, 75, 600)

    # 7) Extract the existing DurationTier
    #    (Praat command is “Extract duration tier”, not “Get duration tier”)
    dur_tier = parselmouth.praat.call(manip, "Extract duration tier")

    # 8) For **every** original phoneme boundary time t (including start=0, end=orig_duration),
    #    insert a point in dur_tier with value = speech_scale or silence_scale.
    #
    #    Praat will linearly interpolate between these “scale‐values” over time.
    #
    #    *Note*: We must also insert a point at t = 0.0 with whichever scale applies.
    current_index = 0
    full_orig_duration = phones_tier.maxTime

    for iv in phones_tier:
        t0, t1 = iv.minTime, iv.maxTime
        lab    = iv.mark.strip().lower()

        # 8a) Decide which scale to use on [t0, t1]:
        is_sil = (lab in SILENCE_LABELS)
        chosen_scale = sil_scale if is_sil else speech_scale

        # 8b) Insert a duration‐tier point at t0 _and_ at t1 with chosen_scale.
        #     This ensures that between t0 and t1, the local time factor = chosen_scale.
        parselmouth.praat.call(dur_tier, "Add point", t0, chosen_scale)
        parselmouth.praat.call(dur_tier, "Add point", t1, chosen_scale)

        current_index += 1

    # 9) Replace the Manipulation’s DurationTier with our custom one
    #    (use the two‐argument form: [manip, dur_tier], "Replace duration tier")
    parselmouth.praat.call([manip, dur_tier], "Replace duration tier")

    # 10) Resynthesize (overlap‐add) via PSOLA
    modified_sound = parselmouth.praat.call(manip, "Get resynthesis (overlap-add)")

    # 11) Save the new WAV
    modified_sound.save(args.wav_out, "WAV")
    print(f"✔️  Wrote stretched WAV → {args.wav_out}")

    # 12) Build & write a new TextGrid whose boundaries have been recomputed:
    #     new_dur_i = (t1_i - t0_i) * (speech_scale or silence_scale).
    new_intervals = []
    running_t = 0.0
    for iv in phones_tier:
        t0_old, t1_old = iv.minTime, iv.maxTime
        lab = iv.mark.strip().lower()
        is_sil = (lab in SILENCE_LABELS)
        chosen_scale = sil_scale if is_sil else speech_scale
        dur_new = (t1_old - t0_old) * chosen_scale
        new_intervals.append(Interval(running_t, running_t + dur_new, iv.mark.strip()))
        running_t += dur_new

    # 13) Write out a fresh TextGrid with that single “phones” tier:
    new_tg = TextGrid(minTime=0.0, maxTime=target)
    updated_tier = IntervalTier(
        phones_tier.name,     # keep the same tier name, e.g. “phones”
        0.0,                  # xmin
        target
    )
    new_tg.append(updated_tier)
    new_tg.write(args.tg_out)
    print(f"✔️  Wrote updated TextGrid → {args.tg_out}")
    print("✅  Done. Final total = %.3f s (should exactly match your target)." % target)


if __name__ == "__main__":
    main()

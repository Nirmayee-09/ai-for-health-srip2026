import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import read_signal, read_events

def find_file(folder, *keywords, exclude=None):
    candidates = glob.glob(os.path.join(folder, "*.txt"))
    for path in candidates:
        name_lower = os.path.basename(path).lower()
        if all(kw.lower() in name_lower for kw in keywords):
            if exclude and exclude.lower() in name_lower:
                continue
            return path
    raise FileNotFoundError(
        f"No file matching {keywords} (excluding '{exclude}') in {folder}"
    )

# ── STAGE 1: Filter ─────────────────────────────────────────────
'''
this filter is used to suppress noise
so here :-   (actual data)-(higher freq)-(lower freq) = (actaul brreathing freq)
-----------------------------------------------------------------------------------------------
def bandpass_filter(signal_values, lowcut=0.17, highcut=0.4, fs=32):

signal values     : array of actual data
lowcut            : lowest breathing is 10 breathes per min = 10/60 per second ...approx 0.17Hz
highcut           : highest breathing is 24 breathes per min = 24/60 per second ...approx 0.4Hz
fs(sampling freq) : number of data points are collected per second
nyq(Nyquist freq) : for accurate detection fs should be 2x the highest freq 
                     nyq is exactly at half of fs and thus is the maximum detectable freq
-----------------------------------------------------------------------------------------------
b, a = butter(4, [low, high], btype='band')
butter creates the a and b parameteres that form a diff eqn to get the correct filtered sample

1)    4 is the order... mid level:- 
      if order less - nearby unwanted frequencies are not fully suppressed
      if order more - Steeper cutoff & Sharper separation between kept and removed frequencies

2)    butter needs normalized values (btw 0 and 1 to mark the filter)
      lowcut or highcut/nyq = normalized value... 

3)    btype = which values are allowed .... highpass , lowpass {2hz (0.5 allowed but not 4)}
      bandpass means values between passed cutoff allowed 
-----------------------------------------------------------------------------------------------
return filtfilt(b, a, signal_values)
this actually usesthe butter parameters a and b to perfeom the filtering on passed signal values
-----------------------------------------------------------------------------------------------
'''
def bandpass_filter(signal_values, lowcut=0.17, highcut=0.4, fs=32):
    nyq  = fs / 2
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, signal_values)

# ── STAGE 2: Windowing ───────────────────────────────────────────

def create_windows(flow_df, thorac_df, spo2_df, 
                   window_sec=30, overlap_sec=15):
    
    """
    flow_df, thorac_df, spo2_df : filtered dataframes with DatetimeIndex
    window_sec                  : how long each window is (30 seconds)
    overlap_sec                 : how much windows overlap (15 seconds)

    Returns a list of dicts, one dict per window:
    {
        "start"      : timestamp when window starts,
        "end"        : timestamp when window ends,
        "flow"       : numpy array of 960 (30sec * 32Hz = 960 samples) flow values,
        "thorac"     : numpy array of 960 (30sec * 32Hz = 960 samples) thorac values,
        "spo2"       : numpy array of 120 (30sec * 4Hz = 120 samples) spo2 values
    }
    """

    windows = []
    step   = pd.Timedelta(seconds=overlap_sec)
    window = pd.Timedelta(seconds=window_sec)


    t = flow_df.index[0]
    recording_end = flow_df.index[-1]

    while t + window <= recording_end:
        win_start = t
        win_end   = t + window

        # slice each signal to this 30s chunk
        # .loc on DatetimeIndex gives all rows between win_start and win_end
        flow_win   = flow_df.loc[win_start:win_end]["value"].values
        thorac_win = thorac_df.loc[win_start:win_end]["value"].values
        spo2_win   = spo2_df.loc[win_start:win_end]["value"].values

        windows.append({
            "start"  : win_start,
            "end"    : win_end,
            "flow"   : flow_win,
            "thorac" : thorac_win,
            "spo2"   : spo2_win
        })

        # move forward by 15 seconds (50% overlap)
        t += step

    return windows 

# ── STAGE 3: Labeling ────────────────────────────────────────────

def label_window(win_start, win_end, events_df, window_sec=30):
    """
    win_start, win_end : timestamps of the current window
    events_df          : dataframe from read_events() with start/end/breathing_label
    window_sec         : 30 seconds

    Returns a string label: "Normal", "Hypopnea", "Obstructive Apnea", etc.
    """

    window_duration = pd.Timedelta(seconds=window_sec)
    threshold       = window_duration * 0.5   # 30sec * 0.5(half) = 15 seconds

    # only check events that overlap with this window at all
    candidates = events_df[
        (events_df["start"] < win_end) &
        (events_df["end"]   > win_start)
    ]

    for _, ev in candidates.iterrows():

        # how much does this event overlap with our window?
        overlap_start = max(win_start, ev["start"])
        overlap_end   = min(win_end,   ev["end"])
        overlap_duration = overlap_end - overlap_start

        # if overlap is more than 50% of window → assign this label
        if overlap_duration > threshold:
            return ev["breathing_label"]

    # no event overlapped enough → Normal
    return "Normal"

# ── STAGE 4: main() ──────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir",  required=True)
    parser.add_argument("-out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []

    # step 1: get list of participant folders [Data/AP01, Data/AP02 ...]
    participant_folders = sorted(glob.glob(os.path.join(args.in_dir, "AP*")))

    # step 2: loop over each participant
    for folder in participant_folders:
        participant = os.path.basename(folder)
        print(f"Processing {participant}...")

        # step 3a: load
        flow_df   = read_signal(find_file(folder, "flow", exclude="event"))
        thorac_df = read_signal(find_file(folder, "thorac"))
        spo2_df   = read_signal(find_file(folder, "spo2"))
        events_df = read_events(find_file(folder, "flow", "event"))

        # step 3b: filter
        flow_df["value"]   = bandpass_filter(flow_df["value"].values,   fs=32)
        thorac_df["value"] = bandpass_filter(thorac_df["value"].values, fs=32)
        spo2_df["value"]   = bandpass_filter(spo2_df["value"].values,   fs=4)

        # step 3c: chop into windows
        windows = create_windows(flow_df, thorac_df, spo2_df)

        # step 3d + 3e: label each window and save as row
        for w in windows:
            label = label_window(w["start"], w["end"], events_df)

            # one row = all signal values + label + who it belongs to
            row = {}
            row["participant"] = participant
            row["win_start"]   = w["start"]
            row["win_end"]     = w["end"]
            row["label"]       = label

            # store the actual signal arrays as-is
            # instead of flow_0, flow_1 ... we just store the whole array
            row["flow"]   = w["flow"]
            row["thorac"] = w["thorac"]
            row["spo2"]   = w["spo2"]

            all_rows.append(row)

        print(f"  Done — {len(windows)} windows")

    # step 4: save
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(args.out_dir, "breathing_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
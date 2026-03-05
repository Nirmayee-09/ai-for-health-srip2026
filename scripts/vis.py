import argparse
import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

import matplotlib
# Matplotlib may try to open GUI plot windows everytime we do plt.plot(); with "Agg", it renders plots off-screen in memory(safe and reliable)
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# ── Import your utility functions ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) # _file_ is a variable that by default stores the current file path
# We first search the parent directory of the current file to look for the imports
from utils import read_signal, read_events, read_sleep_profile


# ── STEP 1: Smart file finder ───────────────────────────────────────────────────
# Filenames differ across participants (e.g. "Flow" vs "Flow Nasel").
# Looks at all files... checks if any one has ALL required keywords and Compulsorily Does NOT have Excluded keywords in the name
# If no file from all the files in the given folder match the requirement raise file not found error

def find_file(folder, *keywords, exclude=None):
    # glob takes all files from the mentioned 'folder'(taken as input to the find_file)
    #os.path.join helps me add the folder to the file path of the file... so i get folder/file.txt.... eg:- AP01/flow.txt
    candidates = glob.glob(os.path.join(folder, "*.txt")) #list of all files
    for path in candidates:
        name_lower = os.path.basename(path).lower()
        # Check all keywords are present
        if all(kw.lower() in name_lower for kw in keywords):
            # Check exclusion
            if exclude and exclude.lower() in name_lower:
                continue
            return path
    raise FileNotFoundError(
        f"No file matching keywords {keywords} (excluding '{exclude}') in {folder}"
    )


# ── STEP 2: Colour map for events ──────────────────────────────────────────────
# Each event type gets a distinct transparent (0.45) color for the shaded region.
EVENT_COLORS = {
    "Hypopnea":          ("gold",        0.45),
    "Obstructive Apnea": ("tomato",      0.45),
    "Central Apnea":     ("mediumpurple",0.45),
    "Mixed Apnea":       ("deepskyblue", 0.45),
}
DEFAULT_EVENT_COLOR = ("lightgrey", 0.45)   # fallback for unknown event types


# ── STEP 3: Plot a single 6-minute page ────────────────────────────────────────

def plot_page(ax_flow, ax_thorac, ax_spo2,
              flow_df, thorac_df, spo2_df,
              events_df, window_start, window_end, participant_name):
    """
    ax_flow, ax_thorac, ax_spo2 : matplotlib Axes
    flow_df, thorac_df, spo2_df : DataFrames with DatetimeIndex and 'value' column
    events_df                   : DataFrame with 'start','end','breathing_label' columns
    window_start, window_end    : pd.Timestamp defining the 6-minute slice
    participant_name            : string used in the page title
    """

    # -- Slice each signal to this 6-minute window --
    # .loc on a DatetimeIndex accepts timestamps as slice bounds (inclusive)
    flow_win   = flow_df.loc[window_start:window_end]
    thorac_win = thorac_df.loc[window_start:window_end]
    spo2_win   = spo2_df.loc[window_start:window_end]

    # -- Plot signals -- 
    #indexis timestamp same for all three 
    ax_flow.plot(flow_win.index,   flow_win["value"],
                 color="steelblue", linewidth=0.5, label="Nasal Flow")
    ax_thorac.plot(thorac_win.index, thorac_win["value"],
                   color="darkorange", linewidth=0.5, label="Thoracic/Abdominal Resp.")
    ax_spo2.plot(spo2_win.index,   spo2_win["value"],
                 color="dimgray",  linewidth=0.8, label="SpO2")

    # -- Overlay breathing events --
    # Filter events that overlap with this window at all (start < window_end AND end > window_start)
    page_events = events_df[(events_df["start"] < window_end) & (events_df["end"]   > window_start)]

    legend_patches = {}  # collect unique event types for the legend

    for _, ev in page_events.iterrows():
        # itterrows gives (index,rowdata)
        label = ev["breathing_label"] 
        color, alpha = EVENT_COLORS.get(label, DEFAULT_EVENT_COLOR) # alpha means opacity

        # Event can continue after the visible 6 min or being befor the visible 6min
        # in this case window start will be greater and thus we will start outlining the shaded rectangle form the window start
        # in case the event starts in between and ends in between the ev[start] ev[end] eill be used as marking parameters for the shaded rectangle
        ev_start = max(ev["start"], window_start)
        ev_end   = min(ev["end"],   window_end)

        # axvspan uses the marking parameter ev_start and ev_end & draws a vertical shaded band across the full y-range of an axis as z=2
        ax_flow.axvspan(ev_start, ev_end, color=color, alpha=alpha, zorder=2)

        # Label text on top subplot only (avoids clutter)
        ax_flow.text(
            ev_start, 
            ax_flow.get_ylim()[1] * 0.85,
            label,
            fontsize=5, color="black", rotation=0,
            va="top", ha="left", clip_on=True
        )
            

    # -- Axis labels --
    ax_flow.set_ylabel("Nasal Flow\n(l/min)", fontsize=6)
    ax_thorac.set_ylabel("Resp.\nAmplitude", fontsize=6)
    ax_spo2.set_ylabel("SpO2 (%)", fontsize=6)
    ax_spo2.set_xlabel("Time", fontsize=6)

    # -- Legends (top-right of each subplot) --
    for ax in [ax_flow, ax_thorac, ax_spo2]:
        ax.legend(loc="upper right", fontsize=5)


    # -- Grid --
    for ax in [ax_flow, ax_thorac, ax_spo2]:
        ax.grid(True, alpha=0.5)

    # -- X-axis tick formatting: show HH:MM:SS, rotate for readability --
    import matplotlib.dates as mdates
    # hide x labels on top two plots
    for ax in [ax_flow, ax_thorac]:
        ax.set_xticklabels([])         
    #formats the date and time and makes it in hh:mm:ss format 
    ax_spo2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    # plots all xticls labels at interval of 30 sec
    ax_spo2.xaxis.set_major_locator(mdates.SecondLocator(interval=5))
    # plt.setp() is a dual-purpose function — it can either:
    # Set properties when you pass kwargs → plt.setp(obj, color="red")
    # Prints biggg dump of available properties when you pass nothing → plt.setp(obj)
    plt.setp(ax_spo2.get_xticklabels(),fontsize=5,rotation=90)

    # -- Tick label sizes of x and y axis set to 5 --
    for ax in [ax_flow, ax_thorac, ax_spo2]:
        ax.tick_params(labelsize=5)

    # -- Page title --
    title = (f"{participant_name}  —  "
             f"{window_start.strftime('%Y-%m-%d %H:%M:%S')}  to  "
             f"{window_end.strftime('%Y-%m-%d %H:%M:%S')}")
    ax_flow.set_title(title, fontsize=7)


# ── STEP 4: Main function ───────────────────────────────────────────────────────

def main():
    # -- Parse command-line argument --
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", required=True,
                        help='Path to participant folder, e.g. "Data/AP01"') # python vis.py -h then this msg pops
    # Store the current command line argument in arrgs
    args = parser.parse_args()
    # args.name gives folder name of the directory passed to "-name"
    folder = args.name
    # Extracts the last part of the path e.g. "AP01"
    participant_name = os.path.basename(folder) 

    # -- Find files using keyword matching --
    print(f"[INFO] Looking for signal files in: {folder}")
    flow_path    = find_file(folder, "flow",   exclude="event")
    thorac_path  = find_file(folder, "thorac")
    spo2_path    = find_file(folder, "spo2")
    events_path  = find_file(folder, "flow", "event")
    sleep_path   = find_file(folder, "sleep")
    print(f"  Flow   : {os.path.basename(flow_path)}")
    print(f"  Thorac : {os.path.basename(thorac_path)}")
    print(f"  SpO2   : {os.path.basename(spo2_path)}")
    print(f"  Events : {os.path.basename(events_path)}")
    print(f"  Sleep  : {os.path.basename(sleep_path)}")

    # -- Load data into df --
    print("[INFO] Loading signals...")
    flow_df   = read_signal(flow_path)
    thorac_df = read_signal(thorac_path)
    spo2_df   = read_signal(spo2_path)
    events_df = read_events(events_path)

    # -- Build 6-minute page windows across the full recording --
    # We use the flow signal's timeline as the reference (it's the longest, 32 Hz)
    recording_start = flow_df.index[0]
    recording_end   = flow_df.index[-1]

    # timedelta is a pandas class that represents duration btw 2 times
    window_size = pd.Timedelta(minutes=6)

    #we store the start and end time stamps of each 6 min window in a tuple form in windows list... [(hh:mm:ss,HH:MM:SS),(hh:mm:ss,HH:MM:SS)]
    windows = []
    t = recording_start
    while t < recording_end:
        windows.append((t, t + window_size))
        t += window_size

    print(f"[INFO] Total pages: {len(windows)}")

    # -- Create output directory if it doesn't exist --
    out_dir = "Visualizations"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{participant_name}_visualization.pdf")

    # -- Generate PDF --
    print(f"[INFO] Writing PDF to: {out_path}")
    
    # Open a PDF file that can hold multiple pages
    pdf = PdfPages(out_path)
    
    # Loop through each 6-minute window
    page_num = 1
    for (win_start, win_end) in windows:
        
        # Create a blank figure with 3 subplots stacked vertically
        #we usually do plt.plot(), plt.xtixks(), etc... plt is fig + axes... so it plots on any available axis
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 6), sharex=True)
        ax1 = axes[0]  # top plot    : Nasal Flow
        ax2 = axes[1]  # middle plot : Thoracic
        ax3 = axes[2]  # bottom plot : SpO2
        
        # Fill the 3 plots with data for this time window
        plot_page(ax1, ax2, ax3,
                  flow_df, thorac_df, spo2_df,
                  events_df, win_start, win_end, participant_name)
        
        # Save this figure as one page in the PDF
        pdf.savefig(fig)
        
        # Delete the figure from memory (we have ~60 pages, don't want to keep all in RAM)
        plt.close(fig)
    
    # Close and finalize the PDF file
    pdf.close()
    
    print(f"[INFO] Done. PDF saved → {out_path}")


# name stores _main_ if file same file is running... else it stores the name of the filw that imported it
if __name__ == "__main__":
    main()
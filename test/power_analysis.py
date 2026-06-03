import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import re
import yaml

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = BASE_DIR / 'test results' / 'power tracking'
OUTPUT_DIR = RESULTS_ROOT / 'comparative_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 20.0
DT = 1.0 / FPS

# Global Theme
sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

def parse_folder_name(folder_name):
    if folder_name == 'baseline': return 'Baseline', 0, 0.0 
    match = re.match(r"([a-z]+)_g(\d+)_([\d.]+)Hz", folder_name)
    if match: return match.group(1).capitalize(), int(match.group(2)), float(match.group(3))
    return 'Unknown', 0, 0.0

def load_data(results_root):
    trace_records = []
    scalar_records = []
    electrode_records = []
    
    for video_dir in [d for d in results_root.iterdir() if d.is_dir() and d.name != 'comparative_analysis']:
        video_name = video_dir.name
        
        for raster_dir in [d for d in video_dir.iterdir() if d.is_dir()]:
            raster_type, groups, rate = parse_folder_name(raster_dir.name)
            
            for edge_dir in [d for d in raster_dir.iterdir() if d.is_dir()]:
                edge_method = edge_dir.name
                avg_path = edge_dir / 'average_power_per_electrode.npy'
                inst_path = edge_dir / 'instant_power_per_electrode.npy'
                
                if not avg_path.exists(): continue

                try:
                    avg_data = np.load(avg_path) 
                    if inst_path.exists(): inst_data = np.load(inst_path)
                    else: inst_data = avg_data 
                    
                    n_electrodes = avg_data.shape[1]

                    # A. Traces
                    avg_trace = np.sum(avg_data, axis=1)
                    frames = np.arange(len(avg_trace))
                    time_sec = frames * DT
                    step = 5
                    for t, p_avg in zip(time_sec[::step], avg_trace[::step]):
                        trace_records.append({
                            'Time (s)': t, 'Avg Power (mW)': p_avg,
                            'Raster': raster_type, 'Groups': groups, 'Edge': edge_method, 'Video': video_name
                        })

                    # B. Mean Proportion of Activated Electrodes
                    # Calculate percentage of grid active in each frame, then average over time
                    active_per_frame = np.sum(avg_data > 0, axis=1)
                    pct_per_frame = (active_per_frame / n_electrodes) * 100.0
                    mean_active_pct = np.mean(pct_per_frame)

                    # C. Scalars
                    total_energy_mj = np.sum(avg_data) * DT
                    peak_inst_power = np.max(np.sum(inst_data, axis=1))
                    
                    scalar_records.append({
                        'Raster': raster_type, 
                        'Groups': groups, 
                        'Edge': edge_method, 
                        'Video': video_name,
                        'Electrodes': n_electrodes,
                        'Total Energy (mJ)': total_energy_mj, 
                        'Peak Instant Power (mW)': peak_inst_power,
                        'Mean Active Electrodes (%)': mean_active_pct 
                    })
                    
                    # D. Electrodes
                    electrode_energy = np.sum(avg_data, axis=0) * DT
                    active_energies = electrode_energy[electrode_energy > 0]
                    for energy in active_energies:
                        electrode_records.append({
                            'Raster': raster_type, 'Groups': groups, 'Edge': edge_method, 'Electrode Energy (mJ)': energy
                        })
                    
                    print(f"Loaded: {raster_type} (g={groups}, edge={edge_method}) -> {mean_active_pct:.1f}% Mean Active")

                except Exception as e:
                    print(f"Error processing {edge_dir}: {e}")

    return pd.DataFrame(trace_records), pd.DataFrame(scalar_records), pd.DataFrame(electrode_records)

def save_electrode_counts(df_scalars):
    print("Generating Electrode Count Report...")
    cols = ['Video', 'Raster', 'Groups', 'Edge', 'Electrodes', 'Mean Active Electrodes (%)']
    summary_table = df_scalars[cols].sort_values(by=['Video', 'Raster', 'Groups', 'Edge'])
    csv_path = OUTPUT_DIR / 'electrode_counts.csv'
    summary_table.to_csv(csv_path, index=False)
    print(f"Saved count table to: {csv_path}")

def plot_power_traces(df_traces):
    print("Generating Power Trace plots...")
    edge_methods = df_traces['Edge'].unique()
    for edge in edge_methods:
        subset = df_traces[df_traces['Edge'] == edge]
        plt.figure(figsize=(14, 7))
        
        # Lineplot with markers and no dashes
        ax = sns.lineplot(data=subset, x='Time (s)', y='Avg Power (mW)', 
                     hue='Raster', style='Groups', palette='tab10', linewidth=2.0,
                     markers=True, dashes=False, markersize=9)
        
        plt.title(f"Average Power Trace ({edge.upper()})", fontsize=28, fontweight='bold', pad=20)
        plt.xlabel("Time (s)", fontsize=22) 
        plt.ylabel("Avg Power (mW)", fontsize=22) 
        plt.xticks(fontsize=17) 
        plt.yticks(fontsize=17) 
        
        # Legend styling
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), fontsize=15, title_fontsize=16)
        leg = ax.get_legend()
        if leg:
            for text in leg.get_texts():
                if text.get_text() in ['Raster', 'Groups']:
                    text.set_fontweight('bold')
                    text.set_fontsize(17)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'trace_avg_{edge}.png', dpi=150)
        plt.close()

def plot_bar_comparisons(df_scalars):
    print("Generating Bar Comparisons...")
    df_scalars['Config'] = df_scalars.apply(lambda x: f"{x['Raster']}\n(g={x['Groups']})" if x['Groups'] > 0 else "Baseline", axis=1)
    df_scalars.sort_values(by=['Raster', 'Groups'], inplace=True)
    
    # 1. Total Energy
    plt.figure(figsize=(16, 10))
    sns.barplot(data=df_scalars, x='Config', y='Total Energy (mJ)', hue='Edge', palette='viridis', errorbar=None)
    plt.title("Total Energy Consumption", fontsize=30, fontweight='bold', pad=20)
    plt.xlabel("Configuration", fontsize=24, labelpad=15)
    plt.ylabel("Total Energy (mJ)", fontsize=24)
    plt.xticks(fontsize=17); plt.yticks(fontsize=17)
    plt.legend(title='Preprocessing', title_fontsize=18, fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bar_total_energy.png', dpi=150)
    plt.close()

    # 2. Peak Power
    plt.figure(figsize=(16, 10))
    sns.barplot(data=df_scalars, x='Config', y='Peak Instant Power (mW)', hue='Edge', palette='magma', errorbar=None)
    plt.title("Maximum Hardware Power Spike", fontsize=30, fontweight='bold', pad=20)
    plt.xlabel("Configuration", fontsize=24, labelpad=15)
    plt.ylabel("Peak Power (mW)", fontsize=24)
    plt.xticks(fontsize=17); plt.yticks(fontsize=17)
    plt.legend(title='Preprocessing', title_fontsize=18, fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bar_peak_instant.png', dpi=150)
    plt.close()

def plot_activation_proportion(df_scalars):
    print("Generating Mean Activation Proportion Plot...")
    df_scalars['Config'] = df_scalars.apply(lambda x: f"{x['Raster']}\n(g={x['Groups']})" if x['Groups'] > 0 else "Baseline", axis=1)
    df_scalars.sort_values(by=['Raster', 'Groups'], inplace=True)
    
    plt.figure(figsize=(16, 10))
    sns.barplot(data=df_scalars, x='Config', y='Mean Active Electrodes (%)', hue='Edge', palette='coolwarm', errorbar=None)
    
    plt.title("Mean Proportion of Activated Electrodes", fontsize=30, fontweight='bold', pad=20)
    plt.xlabel("Configuration", fontsize=24, labelpad=15)
    plt.ylabel("Mean Active Electrodes (%)", fontsize=24)
    
    plt.ylim(0, 100)
    plt.xticks(fontsize=17); plt.yticks(fontsize=17)
    plt.legend(title='Preprocessing', title_fontsize=18, fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bar_active_proportion.png', dpi=150)
    plt.close()

def plot_electrode_distributions(df_electrodes):
    print("Generating Electrode Distribution Boxplots...")
    df_electrodes['Config'] = df_electrodes.apply(lambda x: f"{x['Raster']}\n(g={x['Groups']})" if x['Groups'] > 0 else "Baseline", axis=1)
    df_electrodes.sort_values(by=['Raster', 'Groups'], inplace=True)
    
    edge_methods = df_electrodes['Edge'].unique()
    for edge in edge_methods:
        subset = df_electrodes[df_electrodes['Edge'] == edge]
        plt.figure(figsize=(16, 10))
        sns.boxplot(data=subset, x='Config', y='Electrode Energy (mJ)', hue='Config', legend=False, palette='Set3', showfliers=False) 
        
        plt.title(f"Energy Distribution per Electrode ({edge.upper()})", fontsize=30, fontweight='bold', pad=20)
        plt.xlabel("Configuration", fontsize=24, labelpad=15)
        plt.ylabel("Total Energy (mJ)", fontsize=24)
        plt.xticks(fontsize=17); plt.yticks(fontsize=17)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'boxplot_electrodes_{edge}.png', dpi=150)
        plt.close()

def main():
    print("--- STARTING ANALYSIS ---")
    df_traces, df_scalars, df_electrodes = load_data(RESULTS_ROOT)
    
    if df_scalars.empty:
        print("No data found.")
        return

    print(f"Loaded {len(df_scalars)} simulations.")
    
    save_electrode_counts(df_scalars)
    plot_power_traces(df_traces)
    plot_bar_comparisons(df_scalars)
    plot_activation_proportion(df_scalars)
    plot_electrode_distributions(df_electrodes)
    
    print(f"--- DONE. Results in {OUTPUT_DIR} ---")

if __name__ == "__main__":
    main()
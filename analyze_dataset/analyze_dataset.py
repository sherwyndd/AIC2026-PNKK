import os
import glob
import time
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_video_metadata(filepath):
    """Extract basic metadata for a single video file"""
    try:
        filename = os.path.basename(filepath)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        # Group prefix extraction (e.g. K01_V001.mp4 -> K01)
        group = filename.split('_')[0] if '_' in filename else 'Other'

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        duration_sec = total_frames / fps if fps > 0 else 0
        bitrate_kbps = (file_size_mb * 1024 * 8) / duration_sec if duration_sec > 0 else 0

        return {
            "filename": filename,
            "group": group,
            "filepath": filepath,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "file_size_mb": round(file_size_mb, 2),
            "bitrate_kbps": round(bitrate_kbps, 2)
        }
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def analyze_video_dataset(dataset_dir, output_plot="dataset_analysis.png", max_workers=16):
    """
    Scan and analyze overview metrics for all video files in a dataset directory.
    """
    print(f"Scanning video files in directory: {dataset_dir}...")
    
    extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(dataset_dir, ext)))
        video_files.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))

    video_files = sorted(list(set(video_files)))
    total_found = len(video_files)
    print(f"Found total of {total_found} video files.")

    if total_found == 0:
        print("No video files found!")
        return None

    # Multi-processing parallel scan
    start_time = time.time()
    results = []
    
    print(f"Extracting metadata using {max_workers} parallel workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_video_metadata, f): f for f in video_files}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    scan_duration = time.time() - start_time
    print(f"Completed scanning {len(results)} videos in {scan_duration:.2f} seconds.")

    df = pd.DataFrame(results).sort_values(by='filename').reset_index(drop=True)
    
    # Summary Metrics Calculation
    total_videos = len(df)
    total_size_gb = df['file_size_mb'].sum() / 1024
    total_duration_hours = df['duration_sec'].sum() / 3600
    avg_duration_sec = df['duration_sec'].mean()
    min_duration_sec = df['duration_sec'].min()
    max_duration_sec = df['duration_sec'].max()

    print("=" * 65)
    print("         DATASET OVERVIEW & METRICS REPORT")
    print("=" * 65)
    print(f"• Total Videos            : {total_videos} files")
    print(f"• Total Dataset Storage   : {total_size_gb:.2f} GB")
    print(f"• Total Video Duration    : {total_duration_hours:.2f} Hours ({df['duration_sec'].sum() / 60:.1f} Mins)")
    print(f"• Average Duration        : {avg_duration_sec:.2f}s (Min: {min_duration_sec:.2f}s, Max: {max_duration_sec:.2f}s)")
    print(f"• Resolutions Breakdown   : {dict(df['resolution'].value_counts())}")
    print(f"• FPS Breakdown           : {dict(df['fps'].value_counts())}")
    print(f"• Total Groups / Categories: {df['group'].nunique()} groups ({list(df['group'].unique()[:5])}...)")
    print("=" * 65)

    # ------------------- MATPLOTLIB VISUALIZATION -------------------
    fig = plt.figure(figsize=(16, 11))
    plt.suptitle(f"VIDEO DATASET OVERVIEW ANALYSIS ({total_videos} Videos | {total_size_gb:.1f} GB | {total_duration_hours:.1f} Hours)", 
                 fontsize=14, fontweight='bold', y=0.98)

    # 1. Video Duration Distribution (Histogram)
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(df['duration_sec'] / 60, bins=25, color='#3498db', edgecolor='black', alpha=0.8)
    ax1.set_title("1. Video Duration Distribution (Minutes)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Duration (Minutes)")
    ax1.set_ylabel("Number of Videos")
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 2. Resolution Breakdown (Bar Chart)
    ax2 = plt.subplot(2, 3, 2)
    res_counts = df['resolution'].value_counts()
    ax2.bar(res_counts.index, res_counts.values, color='#2ecc71', edgecolor='black', alpha=0.8, width=0.4)
    ax2.set_title("2. Resolution Breakdown", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Resolution")
    ax2.set_ylabel("Number of Videos")
    for i, v in enumerate(res_counts.values):
        ax2.text(i, v + (max(res_counts.values)*0.02), str(v), ha='center', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. Video Count per Group/Prefix (K01, K02...)
    ax3 = plt.subplot(2, 3, 3)
    group_stats = df.groupby('group').agg({'filename': 'count'}).reset_index().sort_values(by='group')
    
    x = np.arange(len(group_stats))
    ax3.bar(x, group_stats['filename'], color='#e74c3c', alpha=0.8, edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(group_stats['group'], rotation=45, ha='right', fontsize=8)
    ax3.set_title("3. Video Count per Group (Prefix)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Group / Category")
    ax3.set_ylabel("Number of Videos")
    ax3.grid(True, linestyle=':', alpha=0.6)

    # 4. File Size vs Duration Scatter Plot
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(df['duration_sec'] / 60, df['file_size_mb'], c=df['bitrate_kbps'], cmap='viridis', alpha=0.7, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Bitrate (kbps)', fontsize=9)
    ax4.set_title("4. File Size (MB) vs Duration (Mins)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Duration (Minutes)")
    ax4.set_ylabel("File Size (MB)")
    ax4.grid(True, linestyle=':', alpha=0.6)

    # 5. Bitrate Distribution (Histogram)
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(df['bitrate_kbps'], bins=25, color='#9b59b6', edgecolor='black', alpha=0.8)
    ax5.set_title("5. Bitrate Distribution (kbps)", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Bitrate (kbps)")
    ax5.set_ylabel("Number of Videos")
    ax5.grid(True, linestyle=':', alpha=0.6)

    # 6. Statistical Summary Box
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = (
        f"===== STATISTICAL SUMMARY =====\n\n"
        f"• Total Videos       : {total_videos}\n"
        f"• Total Size         : {total_size_gb:.2f} GB\n"
        f"• Total Duration     : {total_duration_hours:.2f} Hours\n"
        f"• Avg Duration       : {avg_duration_sec/60:.2f} Mins\n"
        f"• Avg File Size      : {df['file_size_mb'].mean():.1f} MB\n"
        f"• FPS Breakdown      : {dict(df['fps'].value_counts())}\n"
        f"• Avg Bitrate        : {df['bitrate_kbps'].mean():.1f} kbps\n"
        f"• Total Categories   : {df['group'].nunique()}\n"
    )
    ax6.text(0.05, 0.15, summary_text, fontsize=11, family='monospace',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#bdc3c7", alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Chart saved successfully at: {output_plot}")
    
    # Save metadata table to CSV
    csv_path = output_plot.replace('.png', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Exported detailed dataset table to: {csv_path}")

    return df

if __name__ == "__main__":
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "/mlcv2025/Datasets/HCMAI25/batch2/video"
    output_img = sys.argv[2] if len(sys.argv) > 2 else "/AIClub_NAS/core_baotg/phong/AIC_2026/dataset_analysis.png"
    
    analyze_video_dataset(dataset_path, output_plot=output_img, max_workers=16)

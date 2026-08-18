import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.config import get_pipeline_defaults


def load_dataframe(path_csv, path_json):
    if os.path.exists(path_csv):
        return pd.read_csv(path_csv)
    if os.path.exists(path_json):
        return pd.read_json(path_json)
    return None


def build_app(data_dir: str, video_dir: str):
    st.set_page_config(
        page_title="Keyframe & Metadata Inspector",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .title-container {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            padding: 24px;
            border-radius: 12px;
            color: #f9fafb;
            margin-bottom: 25px;
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.05);
        }
        .title-container h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #60a5fa 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .title-container p {
            margin: 8px 0 0 0;
            opacity: 0.8;
            font-size: 1rem;
        }
        .sidebar-header {
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 15px;
            margin-bottom: 10px;
            color: #60a5fa;
            border-bottom: 1px solid rgba(96,165,250,0.2);
            padding-bottom: 5px;
        }
        .kf-label {
            font-size: 0.85rem;
            margin-top: 10px;
            margin-bottom: 10px;
            line-height: 1.4;
            color: #d1d5db;
        }
        .kf-id {
            font-weight: 700;
            color: #60a5fa;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<div class='sidebar-header'>📁 Paths Configuration</div>", unsafe_allow_html=True)
    data_dir = st.sidebar.text_input("Data Directory (output_dir)", value=data_dir)
    video_dir = st.sidebar.text_input("Videos Source Directory", value=video_dir)

    if not os.path.exists(data_dir):
        st.error(f"❌ Data Directory not found: {data_dir}")
        return

    videos_metadata_csv = os.path.join(data_dir, "Metadata", "videos", "videos_metadata.csv")
    videos_metadata_json = os.path.join(data_dir, "Metadata", "videos", "videos_metadata.json")

    df_videos = load_dataframe(videos_metadata_csv, videos_metadata_json)
    if df_videos is None or df_videos.empty:
        st.error("❌ No video metadata found. Ensure Metadata/videos exists.")
        return

    st.sidebar.markdown("<div class='sidebar-header'>🎛️ Control Panel</div>", unsafe_allow_html=True)
    video_ids = df_videos["video_id"].tolist()
    selected_video_id = st.sidebar.selectbox("🎬 Select Video", video_ids)

    video_record = df_videos[df_videos["video_id"] == selected_video_id].iloc[0]

    st.sidebar.markdown("<div class='sidebar-header'>📊 Original Video Info</div>", unsafe_allow_html=True)
    video_info = f"""
    <div style='background-color: rgba(128,128,128,0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.15); font-size: 0.9rem; line-height: 1.5; color: #e5e7eb;'>
        <b>File Name:</b> {video_record.get('file_name', 'N/A')}<br>
        <b>FPS:</b> {video_record.get('fps', 'N/A')}<br>
        <b>Duration:</b> {video_record.get('duration', 'N/A')}s<br>
        <b>Total Frames:</b> {video_record.get('total_frames', 'N/A')}<br>
        <b>Resolution:</b> {video_record.get('width', 'N/A')}x{video_record.get('height', 'N/A')}<br>
    </div>
    """
    st.sidebar.markdown(video_info, unsafe_allow_html=True)

    shots_csv = os.path.join(data_dir, "Metadata", "shots", f"{selected_video_id}_shots.csv")
    shots_json = os.path.join(data_dir, "Metadata", "shots", f"{selected_video_id}_shots.json")
    df_shots = load_dataframe(shots_csv, shots_json)

    kf_csv = os.path.join(data_dir, "Metadata", "keyframes", f"{selected_video_id}_keyframes.csv")
    kf_json = os.path.join(data_dir, "Metadata", "keyframes", f"{selected_video_id}_keyframes.json")
    df_keyframes = load_dataframe(kf_csv, kf_json)

    shot_options = ["All Shots"]
    if df_shots is not None and not df_shots.empty:
        shot_options += [f"{row['shot_id']} ({row['pts_start']:.2f}s - {row['pts_end']:.2f}s)" for _, row in df_shots.iterrows()]

    selected_shot_label = st.sidebar.selectbox("🔍 Filter by Shot", shot_options)
    selected_shot_id = None if selected_shot_label == "All Shots" else selected_shot_label.split()[0]

    grid_cols = st.sidebar.slider("🖼️ Grid Columns", min_value=2, max_value=8, value=4, step=1)

    st.markdown("""
    <div class="title-container">
        <h1>Keyframe & Metadata Inspector</h1>
        <p>Inspect shot boundaries, preview keyframes, and explore metadata interactively.</p>
    </div>
    """, unsafe_allow_html=True)

    video_filename = video_record.get("file_name")
    video_path = None
    if video_filename:
        safe_video_name = os.path.basename(video_filename)
        for candidate in [
            os.path.join(video_dir, safe_video_name),
            os.path.join(data_dir, safe_video_name),
            os.path.join("../", safe_video_name),
        ]:
            if os.path.exists(candidate):
                video_path = candidate
                break

    tab1, tab2 = st.tabs(["🖼️ Keyframe Viewer", "📊 Metadata Explorer"])

    with tab1:
        if video_path:
            if "start_time" not in st.session_state:
                st.session_state.start_time = 0.0
            if "last_video_id" not in st.session_state or st.session_state.last_video_id != selected_video_id:
                st.session_state.last_video_id = selected_video_id
                st.session_state.start_time = 0.0

            st.markdown(f"### 🎥 Video Player: `{os.path.basename(video_path)}`")
            st.video(video_path, start_time=st.session_state.start_time)
            st.markdown("---")
        else:
            st.warning(f"⚠️ Video not found: {video_filename}")
            st.info("Place the source video in the configured video directory.")
            st.markdown("---")

        if df_keyframes is None or df_keyframes.empty:
            st.info("No keyframes metadata found for this video.")
        else:
            filtered_kf = df_keyframes
            if selected_shot_id is not None:
                filtered_kf = df_keyframes[df_keyframes["shot_id"] == selected_shot_id]

            if filtered_kf.empty:
                st.info("No keyframes found for the selected shot.")
            else:
                st.markdown(f"### 🖼️ Keyframes ({len(filtered_kf)})")
                cols = st.columns(grid_cols)
                for idx, (_, kf) in enumerate(filtered_kf.iterrows()):
                    col_idx = idx % grid_cols
                    with cols[col_idx]:
                        img_rel_path = str(kf.get("image_path", ""))
                        safe_path = os.path.normpath(img_rel_path).lstrip("/")
                        if ".." in safe_path.split(os.path.sep):
                            st.error("Access Denied: invalid image path.")
                            continue

                        img_abs_path = os.path.join(data_dir, safe_path)
                        if os.path.exists(img_abs_path):
                            st.image(img_abs_path)
                        else:
                            st.error(f"Image not found: {img_rel_path}")

                        pts_time = float(kf.get("pts_time", 0.0))
                        st.markdown(f"<div class='kf-label'><span class='kf-id'>{kf.get('keyframe_id', 'N/A')}</span><br>🎬 <b>Shot:</b> {kf.get('shot_id', 'N/A')}<br>⏱️ <b>Time:</b> {pts_time:.2f}s<br>🎞️ <b>Frame:</b> {kf.get('frame_idx', 'N/A')}</div>", unsafe_allow_html=True)
                        if video_path and st.button("⏱️ Jump to Video", key=f"jump_{idx}"):
                            st.session_state.start_time = pts_time
                            st.rerun()

    with tab2:
        st.markdown(f"#### 🎞️ Shots Mapping (`{selected_video_id}`)")
        if df_shots is not None and not df_shots.empty:
            st.dataframe(df_shots, hide_index=True)
        else:
            st.info("No shot metadata available.")

        st.markdown(f"#### 🖼️ Keyframes Mapping (`{selected_video_id}`)")
        if df_keyframes is not None and not df_keyframes.empty:
            st.dataframe(df_keyframes, hide_index=True)
        else:
            st.info("No keyframe metadata available.")


def main():
    defaults = get_pipeline_defaults()
    parser = argparse.ArgumentParser(description="Keyframe & Metadata Inspector GUI")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(ROOT / "dataset" / "demo"),
        help="Pipeline demo output (Metadata/ + Keyframes/)",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=str(defaults["input_dir"]),
        help="Directory containing original source videos",
    )
    args, _ = parser.parse_known_args()
    build_app(args.data_dir, args.video_dir)


if __name__ == "__main__":
    main()

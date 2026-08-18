# Implementation Plan: Keyframe & Metadata Inspector GUI

We will create a Streamlit web GUI (`app_demo.py`) that allows users to interactively inspect the results of the Shot Segmentation and Keyframe Extraction pipeline. The app will read metadata (supporting both CSV and JSON formats) and display keyframe grids, player integration with jump-to-time functionality, and interactive data tables.

## Proposed Changes

### [NEW] [app_demo.py](file:///AIClub_NAS/core_baotg/phong/AIC_2026/app_demo.py)
Create the main Streamlit application script containing:
- **CLI and Sidebar configuration**: Support `--data_dir` and `--video_dir` arguments. Provide interactive text fields in the sidebar to override paths if necessary.
- **Robust Metadata Loading**:
  - Load video lists from `videos_metadata.csv` or `videos_metadata.json`.
  - Dynamically load `[video_id]_shots` and `[video_id]_keyframes` in either CSV or JSON formats.
- **Sidebar Details**:
  - Select video dropdown.
  - Video info card (filename, FPS, duration, total frames, resolution, and extra parameters).
  - Select shot filter (All Shots or individual shot IDs).
  - Grid size slider (number of columns for keyframes layout, e.g., 2 to 8 columns).
- **Tab 1: Keyframe Viewer**:
  - Video player with search resolution (checks if the video file exists in `--video_dir`, fallback search locations, or warns if not found).
  - Handles "Jump to Video" event: Clicking a button on a keyframe card seeks the video player to that keyframe's `pts_time` by resetting session state and updating player's `start_time` and element key.
  - Keyframe Grid: Renders cropped/thumbnail images dynamically with clear labels: Keyframe ID, Frame Index, Timestamp. Contains path-traversal sanitization.
- **Tab 2: Metadata Explorer**:
  - Displays interactive pandas DataFrames for the current video's Shots and Keyframe mappings, with native Streamlit search, filter, and download features.
- **Aesthetic Enhancements**:
  - Modern look using transparent cards, hover zoom effects, and micro-interactions designed using custom CSS style injections.

---

## Verification Plan

### Automated/Local Tests
- Run streamlit locally using a custom port and binding to localhost to inspect changes:
  ```bash
  .venv/bin/streamlit run app_demo.py --server.port 8501 --server.address 127.0.0.1 -- --data_dir ./output_test --video_dir ./temp_videos
  ```
- Use curl/browser agent to verify that the app loads.

### Manual Verification
- Verify dropdown selections dynamically filter the lists.
- Click "Jump to Video" on a keyframe to verify the player seeks to the target timestamp.
- Toggle between "All Shots" and specific shots, validating that only matching keyframes appear.
- View Tab 2 and verify data tables match the source JSON files in `output_test`.

### Security Verification
- **Path Traversal Prevention**: Verify that keyframe image paths and video paths are properly checked and sanitized to prevent reading arbitrary system files outside of designated directories.
- **Host Binding**: Bind the Streamlit server to `127.0.0.1` during testing to ensure it is not publicly accessible on `0.0.0.0`.
- **Framework Sanitization**: Leverage Streamlit's built-in components (`st.image`, `st.video`, `st.dataframe`) which sanitize and escape HTML automatically, avoiding raw html injection of user variables.

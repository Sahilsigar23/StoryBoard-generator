"""
StoryBoard Generator - Web UI
A simple Streamlit interface to generate storyboards from text stories
"""

import streamlit as st
import os
import sys
import pandas as pd
from pathlib import Path

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import modules
from utils import read_data
from segmentation import generateSegments, generateSummaries, summarize_with_pegasus
from image_generation import generate_images

# Page configuration
st.set_page_config(
    page_title="StoryBoard Generator",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title("🎬 StoryBoard Generator")
st.markdown("""
Convert your text stories into visual storyboards using AI!
This app will:
1. Break your story into scenes
2. Summarize each scene
3. Generate images for each scene
""")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    summarizer_model = st.selectbox(
        "Summarization Model",
        ["Pegasus (Recommended)", "DistilBART (Faster)"],
        help="Pegasus is more accurate, DistilBART is faster"
    )
    
    # Story type
    story_type = st.selectbox(
        "Story Type",
        ["fairy_tale", "short_story", "custom"],
        help="Used for organizing output files"
    )
    
    # GPU info
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            gpu_name = torch.cuda.get_device_name(0)
            st.success(f"✅ GPU Detected: {gpu_name}")
        else:
            st.warning("⚠️ No GPU detected - Image generation will be slow")
    except:
        st.info("ℹ️ GPU status unknown")

# Main content area
tab1, tab2 = st.tabs(["📝 Enter Story", "📁 Batch Process"])

with tab1:
    st.header("Single Story Generation")
    
    # Text input
    story_text = st.text_area(
        "Enter your story:",
        height=300,
        placeholder="Once upon a time...",
        help="Paste your story text here"
    )
    
    story_name = st.text_input(
        "Story Name:",
        value="my_story",
        help="Name for the output folder"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        generate_btn = st.button("🎨 Generate Storyboard", type="primary", use_container_width=True)
    
    if generate_btn:
        if not story_text.strip():
            st.error("❌ Please enter a story!")
        else:
            # Create progress container
            progress_container = st.container()
            
            with progress_container:
                try:
                    # Sanitize story name again to ensure consistency
                    import re
                    story_name_clean = re.sub(r'[^\w\s-]', '', story_name).strip().replace(' ', '_')
                    if not story_name_clean:
                        story_name_clean = "untitled_story"
                    
                    # Create a temporary dataframe with sanitized values
                    df = pd.DataFrame({
                        'original_story': [story_text],
                        'story_name': [story_name_clean],
                        'story_type': [story_type]
                    })
                    
                    # Step 1: Segmentation
                    with st.spinner("📝 Step 1/4: Segmenting story into scenes..."):
                        df = generateSegments(df)
                        st.success("✅ Segmentation complete!")
                    
                    # Check if segmentation worked
                    segments = df.iloc[0]['segmented_story']
                    if isinstance(segments, str):
                        import ast
                        try:
                            segments = ast.literal_eval(segments)
                        except:
                            segments = []
                    
                    if not segments or len(segments) == 0:
                        st.warning("⚠️ Segmentation didn't create segments. Using fallback method...")
                    elif len(segments) == 1:
                        st.warning(f"⚠️ Only 1 segment created. For multiple images, your story should have multiple scenes/paragraphs.")
                        st.info(f"💡 Tip: Add paragraph breaks or longer text to create multiple scenes.")
                    else:
                        st.success(f"✅ Created {len(segments)} story segments!")
                    
                    # Step 2: Summarization
                    with st.spinner("📝 Step 2/4: Summarizing scenes..."):
                        if summarizer_model == "Pegasus (Recommended)":
                            df = summarize_with_pegasus(df)
                        else:
                            summarizer_model_id = "sshleifer/distilbart-xsum-12-6"
                            df = generateSummaries(df, summarizer_model_id)
                        st.success("✅ Summarization complete!")
                    
                    # Get summaries
                    summaries = df.iloc[0]['summarized_story']
                    if isinstance(summaries, str):
                        import ast
                        try:
                            summaries = ast.literal_eval(summaries)
                        except:
                            summaries = []
                    
                    # Warn if only 1 summary (will result in only 1 image)
                    if summaries and len(summaries) == 1:
                        st.warning(f"⚠️ Only 1 summary created. This will generate only 1 image.")
                        st.info(f"💡 Your story will be split into multiple scenes if it has multiple paragraphs or is longer.")
                    
                    # Show segments and summaries before image generation
                    if segments and summaries:
                        st.subheader("📖 Story Breakdown")
                        for i, (seg, summ) in enumerate(zip(segments, summaries)):
                            with st.expander(f"Scene {i+1}: {summ[:60] if isinstance(summ, str) else str(summ)[:60]}..."):
                                st.write("**Original Segment:**", seg[:500] + "..." if len(str(seg)) > 500 else seg)
                                st.write("**Summary (Image Prompt):**", summ)
                    
                    # Step 3: Generate images
                    st.info("🎨 Step 3/4: Generating images (this may take a while, especially on first run)...")
                    # Set output directory - generate_images will add story_type/story_name if needed
                    output_dir = os.path.join("generated_images", "ui_output")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create progress bar for image generation
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Generate images with progress updates
                    try:
                        generate_images(df, output_dir)
                        progress_bar.progress(100)
                        status_text.text("✅ Image generation complete!")
                    except Exception as img_error:
                        st.error(f"Image generation error: {str(img_error)}")
                        raise
                    
                    # Step 4: Display results
                    st.success("✅ Storyboard generated successfully!")
                    
                    # Display generated images
                    st.subheader("🖼️ Generated Storyboard Images")
                    
                    # Get current story_type and story_name from the dataframe (they might have been sanitized)
                    current_story_type = df.iloc[0]['story_type']
                    current_story_name = df.iloc[0]['story_name']
                    
                    # Images are saved to: output_dir/story_type/story_name/
                    story_image_dir = os.path.join(output_dir, current_story_type, current_story_name)
                    
                    # Also check for nested directory (in case of previous bug)
                    story_image_dir_nested = os.path.join(output_dir, current_story_type, current_story_name, current_story_type, current_story_name)
                    
                    # Try to find images in the correct location
                    image_files = []
                    actual_image_dir = None
                    
                    # First, try the correct location
                    if os.path.exists(story_image_dir):
                        potential_files = [f for f in os.listdir(story_image_dir) if f.endswith('.jpg')]
                        if potential_files:
                            image_files = sorted(potential_files)
                            actual_image_dir = story_image_dir
                    
                    # If not found, try nested location (legacy)
                    if not image_files and os.path.exists(story_image_dir_nested):
                        potential_files = [f for f in os.listdir(story_image_dir_nested) if f.endswith('.jpg')]
                        if potential_files:
                            image_files = sorted(potential_files)
                            actual_image_dir = story_image_dir_nested
                            st.warning("⚠️ Found images in nested directory (legacy structure). Consider regenerating.")
                    
                    # If still not found, search recursively
                    if not image_files:
                        import glob
                        search_pattern = os.path.join(output_dir, current_story_type, current_story_name, "**", "*.jpg")
                        found_files = glob.glob(search_pattern, recursive=True)
                        if found_files:
                            # Extract just the filenames and directory
                            found_files = sorted(found_files)
                            image_files = [os.path.basename(f) for f in found_files]
                            actual_image_dir = os.path.dirname(found_files[0]) if found_files else None
                            if actual_image_dir:
                                st.info(f"📂 Found images in: `{actual_image_dir}`")
                    
                    if image_files and actual_image_dir:
                        # Display images in a grid
                        num_cols = min(3, len(image_files))
                        cols = st.columns(num_cols)
                        for idx, img_file in enumerate(image_files):
                            img_path = os.path.join(actual_image_dir, img_file)
                            # Handle scene number extraction more robustly
                            try:
                                scene_num = int(img_file.replace('segment_', '').replace('.jpg', ''))
                            except:
                                scene_num = idx
                            
                            with cols[idx % num_cols]:
                                try:
                                    st.image(img_path, caption=f"Scene {scene_num + 1}", use_container_width=True)
                                    # Show download button
                                    with open(img_path, "rb") as file:
                                        st.download_button(
                                            label=f"Download Scene {scene_num + 1}",
                                            data=file,
                                            file_name=img_file,
                                            mime="image/jpeg",
                                            key=f"download_{idx}_{current_story_name}"
                                        )
                                except Exception as e:
                                    st.error(f"Error loading image {img_file}: {str(e)}")
                        
                        st.info(f"📁 All images saved to: `{actual_image_dir}`")
                        st.info(f"📝 Story Type: `{current_story_type}` | Story Name: `{current_story_name}`")
                    else:
                        st.warning("⚠️ No images were generated. Check the console/terminal for error messages.")
                        st.info(f"💡 Expected location: `{story_image_dir}`")
                        # Debug: show what directories exist
                        if os.path.exists(output_dir):
                            st.info(f"📂 Output directory exists: `{output_dir}`")
                            try:
                                # Show available story types
                                story_types = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
                                if story_types:
                                    st.info(f"📂 Available story types: {', '.join(story_types)}")
                                    # Show stories for current type
                                    if current_story_type in story_types:
                                        story_type_dir = os.path.join(output_dir, current_story_type)
                                        stories = [d for d in os.listdir(story_type_dir) if os.path.isdir(os.path.join(story_type_dir, d))]
                                        if stories:
                                            st.info(f"📂 Stories in '{current_story_type}': {', '.join(stories)}")
                            except Exception as e:
                                st.debug(f"Debug error: {e}")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        st.exception(e)
                    st.info("💡 Check the terminal/console for more details")

with tab2:
    st.header("Batch Process Stories")
    st.info("Process multiple stories from the data folder")
    
    data_folder = st.text_input(
        "Data Folder:",
        value="data/clean_dataset",
        help="Path to folder containing story subfolders"
    )
    
    if st.button("🚀 Process All Stories", type="primary"):
        if not os.path.exists(data_folder):
            st.error(f"❌ Folder not found: {data_folder}")
        else:
            with st.spinner("🔄 Processing all stories (this will take a while)..."):
                try:
                    # Load data
                    st.info("📚 Loading stories...")
                    df = read_data(data_folder)
                    st.success(f"✅ Loaded {len(df)} stories")
                    
                    # Segmentation
                    st.info("✂️ Segmenting stories...")
                    df = generateSegments(df)
                    
                    # Summarization
                    st.info("📝 Summarizing stories...")
                    if summarizer_model == "Pegasus (Recommended)":
                        df = summarize_with_pegasus(df)
                    else:
                        summarizer_model_id = "sshleifer/distilbart-xsum-12-6"
                        df = generateSummaries(df, summarizer_model_id)
                    
                    # Generate images
                    st.info("🎨 Generating images (this will take a LONG time)...")
                    output_dir = os.path.join("generated_images", "ui_batch_output")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Save CSV first
                    csv_path = os.path.join(output_dir, "df_summary.csv")
                    df.to_csv(csv_path)
                    st.success(f"✅ Saved summaries to {csv_path}")
                    
                    # Generate images
                    generate_images(df, output_dir)
                    
                    st.success("✅ All stories processed!")
                    st.info(f"📁 Output saved to: {output_dir}")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>StoryBoard Generator | Powered by Stable Diffusion 2 & Transformers</p>
</div>
""", unsafe_allow_html=True)


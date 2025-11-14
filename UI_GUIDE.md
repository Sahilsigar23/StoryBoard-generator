# 🎬 StoryBoard Generator - Web UI Guide

## 🚀 How to Start the UI

1. **Open Terminal/Command Prompt** in the project folder

2. **Run the command:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** - Streamlit will automatically open:
   - URL: `http://localhost:8501`
   - Or manually go to: `http://localhost:8501`

---

## 📝 How to Use

### **Tab 1: Enter Story** (Single Story)

1. **Paste your story** in the text area
   - Example: "Once upon a time, there was a princess..."

2. **Enter a story name** (e.g., "my_fairy_tale")

3. **Click "🎨 Generate Storyboard"**

4. **Wait for processing:**
   - Step 1: Segmenting story
   - Step 2: Summarizing segments
   - Step 3: Generating images (takes time!)
   - Step 4: Display results

5. **View results:**
   - See segments and summaries
   - See generated images displayed in the browser

### **Tab 2: Batch Process** (Multiple Stories)

1. **Enter data folder path** (default: `data/clean_dataset`)

2. **Click "🚀 Process All Stories"**

3. **Wait for batch processing** (this takes a LONG time!)

4. **Results saved to:** `generated_images/ui_batch_output/`

---

## ⚙️ Settings (Sidebar)

- **Summarization Model:**
  - **Pegasus (Recommended)** - More accurate summaries
  - **DistilBART (Faster)** - Faster processing

- **Story Type:**
  - `fairy_tale` - For fairy tales
  - `short_story` - For short stories
  - `custom` - For custom stories

- **GPU Status:**
  - Shows if GPU is detected
  - GPU = Fast image generation (~10-30 sec/image)
  - CPU = Slow image generation (~5-10 min/image)

---

## 📤 Output Locations

### Single Story:
- **Images:** `generated_images/ui_output/[story_type]/[story_name]/segment_X.jpg`
- **CSV:** Not saved (only displayed in UI)

### Batch Process:
- **Images:** `generated_images/ui_batch_output/[story_type]/[story_name]/segment_X.jpg`
- **CSV:** `generated_images/ui_batch_output/df_summary.csv`

---

## 🎨 Features

✅ **Text Input** - Paste any story text
✅ **Real-time Progress** - See what's happening
✅ **Image Display** - View generated images in browser
✅ **Segments & Summaries** - See how story was broken down
✅ **GPU Support** - Automatically uses your GPU if available
✅ **Batch Processing** - Process multiple stories at once

---

## ⚠️ Important Notes

1. **First Run:**
   - Will download Stable Diffusion model (~5GB)
   - Takes 10-20 minutes to download
   - Subsequent runs are faster

2. **Image Generation Time:**
   - **With GPU:** ~10-30 seconds per image
   - **Without GPU:** ~5-10 minutes per image

3. **Processing Steps:**
   - Segmentation: Fast (~2-5 sec per story)
   - Summarization: Fast (~1-2 sec per segment)
   - Image Generation: Slow (depends on GPU/CPU)

4. **Browser:**
   - Keep the browser tab open while processing
   - Don't close the terminal/command prompt

---

## 🐛 Troubleshooting

**Problem:** UI won't start
- **Solution:** Make sure you're in the project root folder
- **Solution:** Check if Streamlit is installed: `pip install streamlit`

**Problem:** Import errors
- **Solution:** Make sure all dependencies are installed
- **Solution:** Run from project root: `streamlit run app.py`

**Problem:** Images not generating
- **Solution:** Check GPU status in sidebar
- **Solution:** Check console for error messages
- **Solution:** Make sure models are downloaded (first run)

**Problem:** Browser shows "Connection refused"
- **Solution:** Check if Streamlit is running in terminal
- **Solution:** Try refreshing the page
- **Solution:** Check if port 8501 is available

---

## 🎯 Quick Start Example

1. **Start UI:**
   ```bash
   streamlit run app.py
   ```

2. **In the browser:**
   - Go to "Enter Story" tab
   - Paste: "Three women were changed into flowers which grew in the field..."
   - Name: "test_story"
   - Click "Generate Storyboard"

3. **Wait** (first image takes ~30 seconds with GPU)

4. **See results** - Images appear in the browser!

---

## 📞 Need Help?

- Check the console/terminal for error messages
- Make sure GPU is detected (check sidebar)
- Verify all dependencies are installed
- Check `HOW_IT_WORKS.md` for technical details


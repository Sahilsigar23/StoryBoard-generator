# 🚀 Quick Start Guide - StoryBoard Generator UI

## How to Run the UI

### **Option 1: Using Batch File (Windows)**
1. Double-click `run_ui.bat`
2. Browser will open automatically at `http://localhost:8501`

### **Option 2: Using Command Line**
```bash
streamlit run app.py
```

### **Option 3: Using Python**
```bash
python -m streamlit run app.py
```

---

## 📝 How to Use

### **Step 1: Open the UI**
- Browser opens at: `http://localhost:8501`
- If not, manually go to: `http://localhost:8501`

### **Step 2: Enter Your Story**
1. Go to **"📝 Enter Story"** tab
2. Paste your story in the text area
3. Enter a story name (e.g., "my_story")
4. Click **"🎨 Generate Storyboard"**

### **Step 3: Wait for Processing**
The app will:
1. ✅ Segment your story into scenes
2. ✅ Summarize each scene
3. ✅ Generate images (takes time - first run downloads ~5GB model)
4. ✅ Display images in browser

### **Step 4: View Results**
- See story breakdown (segments & summaries)
- View generated images in a grid
- Download individual images

---

## 🎯 Example Story to Test

Paste this in the text area:

```
Three women were changed into flowers which grew in the field, but one of them was allowed to be in her own home at night. Then once when day was drawing near, and she was forced to go back to her companions in the field and become a flower again, she said to her husband, 'If thou wilt come this afternoon and gather me, I shall be set free, and henceforth will stay with thee.' And he did so.
```

**Story Name:** `test_story`

Click **"Generate Storyboard"** and wait!

---

## ⚙️ Settings (Sidebar)

- **Summarization Model:**
  - Pegasus (Recommended) - More accurate
  - DistilBART (Faster) - Faster processing

- **Story Type:**
  - fairy_tale
  - short_story
  - custom

- **GPU Status:**
  - Shows if GPU is detected
  - GPU = Fast (~10-30 sec/image)
  - CPU = Slow (~5-10 min/image)

---

## 📤 Output Location

Images are saved to:
```
generated_images/ui_output/[story_type]/[story_name]/segment_X.jpg
```

---

## ⚠️ Important Notes

1. **First Run:**
   - Downloads Stable Diffusion model (~5GB)
   - Takes 10-20 minutes
   - Requires internet connection

2. **Image Generation:**
   - With GPU: ~10-30 seconds per image
   - Without GPU: ~5-10 minutes per image

3. **Keep Browser Open:**
   - Don't close the browser tab
   - Don't close the terminal/command prompt

---

## 🐛 Troubleshooting

**UI won't start:**
- Make sure you're in project root folder
- Check if Streamlit is installed: `pip install streamlit`

**Images not generating:**
- Check GPU status in sidebar
- Check terminal for error messages
- First run needs internet for model download

**Browser shows errors:**
- Check terminal/console for details
- Make sure all dependencies are installed

---

## ✅ Ready to Go!

1. Run: `streamlit run app.py`
2. Open browser: `http://localhost:8501`
3. Paste story and click "Generate Storyboard"
4. Wait and see your storyboard!


# 📍 Where Stories Come From & Where Images Go

## 📥 **INPUT: Where Stories Are Read From**

### Current Setup (Command-Line):
The app reads stories from a **folder structure**:

```
data/
└── clean_dataset/
    ├── fairy_tale/
    │   ├── cinderella/
    │   │   └── story.txt  ← Story text file
    │   ├── snow white/
    │   │   └── story.txt
    │   └── ...
    └── short_story/
        ├── story_name_1/
        │   └── story.txt
        └── ...
```

**Location in code:**
- File: `src/main.py` (line 15)
- Path: `data/clean_dataset`
- Function: `read_data()` in `src/utils.py`

**How it works:**
1. Scans `data/clean_dataset/` folder
2. Finds subfolders (story types: `fairy_tale`, `short_story`)
3. Inside each type folder, finds story folders
4. Reads `story.txt` from each story folder
5. Loads all stories into a DataFrame

**Example story file:**
- Path: `data/clean_dataset/fairy_tale/cinderella/story.txt`
- Contains: Full story text (e.g., "The wife of a rich man fell sick...")

---

## 📤 **OUTPUT: Where Images Are Saved**

### Current Setup:
Images are saved to:

```
generated_images/
└── baseline2_try/
    ├── fairy_tale/
    │   ├── cinderella/
    │   │   ├── segment_0.jpg  ← First scene image
    │   │   ├── segment_1.jpg  ← Second scene image
    │   │   └── segment_2.jpg  ← Third scene image
    │   └── snow white/
    │       └── segment_0.jpg
    └── short_story/
        └── [story_name]/
            └── segment_X.jpg
```

**Location in code:**
- File: `src/main.py` (line 16)
- Path: `generated_images/baseline2_try`
- Function: `generate_images()` in `src/image_generation.py` (line 81)

**How it works:**
1. Creates folder structure: `story_type/story_name/`
2. Saves each scene as: `segment_0.jpg`, `segment_1.jpg`, etc.
3. Images are JPG format (512x512 or 768x768 pixels)

**Also generates CSV files:**
- `generated_images/baseline2_try/df_summary_distilbart.csv`
- `generated_images/baseline2_try/df_summary_pegasus.csv`

---

## 🖥️ **Current Status: NO UI**

**Currently, there's NO user interface to:**
- ❌ Paste/upload stories
- ❌ View generated images in browser
- ❌ See progress in real-time

**You have to:**
1. Put story files in `data/clean_dataset/` folder structure
2. Run `python src/main.py` from command line
3. Manually open image files from `generated_images/` folder

---

## 🎨 **Solution: I Created a Web UI!**

I just created `app.py` - a **Streamlit web interface** that will:

✅ **Input:**
- Text area to paste your story
- File upload option
- Batch processing from folder

✅ **Output:**
- Shows generated images directly in browser
- Displays segments and summaries
- Real-time progress updates

**To use it:**
1. Install Streamlit: `pip install streamlit`
2. Run: `streamlit run app.py`
3. Open browser to `http://localhost:8501`

---

## 📊 **Complete Flow Diagram**

```
INPUT (Stories)
    ↓
[data/clean_dataset/]
    ├── fairy_tale/
    │   └── cinderella/story.txt
    └── short_story/
        └── story_name/story.txt
    ↓
[Processing]
    ├── Segmentation
    ├── Summarization
    └── Image Generation
    ↓
OUTPUT (Images)
    ↓
[generated_images/baseline2_try/]
    ├── fairy_tale/
    │   └── cinderella/
    │       ├── segment_0.jpg
    │       ├── segment_1.jpg
    │       └── segment_2.jpg
    └── CSV files with summaries
```

---

## 🔍 **How to View Generated Images**

### Option 1: File Explorer
1. Navigate to: `generated_images/baseline2_try/`
2. Open story folders
3. Double-click `.jpg` files to view

### Option 2: Python Script
```python
from PIL import Image
import os

img_path = "generated_images/baseline2_try/fairy_tale/cinderella/segment_0.jpg"
img = Image.open(img_path)
img.show()
```

### Option 3: Use the Web UI (after installing)
- Images display automatically in browser
- No need to navigate folders

---

## 📝 **Summary**

| Aspect | Current Location |
|--------|-----------------|
| **Input Stories** | `data/clean_dataset/[type]/[story]/story.txt` |
| **Output Images** | `generated_images/baseline2_try/[type]/[story]/segment_X.jpg` |
| **Output CSVs** | `generated_images/baseline2_try/df_summary_*.csv` |
| **View Images** | Manual (open files) or Web UI (after setup) |


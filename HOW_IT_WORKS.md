# 📖 StoryBoard Generator - How It Works

## 🎯 What This App Does

This app automatically converts **text stories** into **visual storyboards** (sequences of images) using AI. It takes a story, breaks it into scenes, summarizes each scene, and generates images for each scene.

---

## 🔄 How It Works (Step-by-Step)

### **Step 1: Load Stories** 📚
- Reads story text files from `data/clean_dataset/`
- Supports multiple story types (fairy tales, short stories, etc.)
- **Input**: Text files (`.txt` files)
- **Output**: DataFrame with story data

### **Step 2: Segmentation** ✂️
**Purpose**: Break long stories into smaller scenes/segments

**What it does**:
- Uses **TextTilingTokenizer** (NLTK) to find natural breaks in the story
- If that fails (short stories), falls back to splitting by sentences
- Groups sentences into chunks of 5 sentences each
- **Input**: Full story text
- **Output**: List of story segments (e.g., ["Scene 1", "Scene 2", ...])

**Technologies**:
- `nltk.tokenize.TextTilingTokenizer` - For topic-based segmentation
- `nltk.tokenize.sent_tokenize` - For sentence splitting

### **Step 3: Summarization** 📝
**Purpose**: Create short, image-generation-friendly prompts from each segment

**What it does**:
- Takes each segment and creates a concise summary (1-2 sentences)
- Uses two models (runs both):
  1. **DistilBART-xsum** - Fast, lightweight summarizer
  2. **Pegasus-xsum** - More accurate summarizer
- **Input**: Story segments
- **Output**: Summarized prompts (e.g., ["A woman turns into a flower", "Her husband rescues her"])

**Technologies**:
- `transformers` library (Hugging Face)
- `sshleifer/distilbart-xsum-12-6` model
- `google/pegasus-xsum` model

### **Step 4: Image Generation** 🎨
**Purpose**: Generate visual storyboard images from text prompts

**What it does**:
1. **First Image**: Uses **Stable Diffusion 2** to generate from the first prompt
2. **Subsequent Images**: Uses **Img2Img** (image-to-image) to generate remaining images, conditioning each new image on the previous one for visual consistency
3. Saves images in organized folders

**Technologies**:
- `diffusers` library (Hugging Face)
- `stabilityai/stable-diffusion-2` - Text-to-image model (~5GB)
- `StableDiffusionImg2ImgPipeline` - Image-to-image model
- `bert-base-uncased` - For finding similar prompts (advanced feature)

**Hardware**:
- **GPU** (NVIDIA GeForce GTX 1650) - Fast generation (~10-30 sec/image)
- **CPU** - Fallback, much slower (~5-10 min/image)

---

## 🛠️ Technologies Used

### **Core Libraries**:
1. **PyTorch** - Deep learning framework
2. **Transformers** (Hugging Face) - Pre-trained NLP models
3. **Diffusers** (Hugging Face) - Image generation models
4. **NLTK** - Natural language processing (tokenization, segmentation)
5. **Pandas** - Data manipulation
6. **Pillow (PIL)** - Image processing

### **AI Models**:
1. **TextTilingTokenizer** - Story segmentation
2. **DistilBART-xsum** - Fast summarization
3. **Pegasus-xsum** - Accurate summarization
4. **Stable Diffusion 2** - Image generation
5. **BERT** - Text similarity (for advanced features)

---

## 📤 Output Structure

### **CSV Files** (in `generated_images/baseline2_try/`):
- `df_summary_distilbart.csv` - Summaries using DistilBART
- `df_summary_pegasus.csv` - Summaries using Pegasus

### **Image Folders** (in `generated_images/baseline2_try/`):
```
generated_images/baseline2_try/
├── fairy_tale/
│   ├── cinderella/
│   │   ├── segment_0.jpg  (First scene)
│   │   ├── segment_1.jpg  (Second scene)
│   │   └── segment_2.jpg  (Third scene)
│   └── snow white/
│       ├── segment_0.jpg
│       └── segment_1.jpg
└── short_story/
    └── [story_name]/
        └── segment_X.jpg
```

### **What Each Output Contains**:
- **CSV Files**: 
  - Original story text
  - Segmented story (list of scenes)
  - Summarized story (list of prompts)
  - Story metadata (name, type)

- **Images**:
  - JPG files (512x512 or 768x768 pixels)
  - Named by segment number
  - Organized by story type and story name

---

## 🎬 Example Workflow

**Input Story**:
```
"Three women were changed into flowers which grew in the field, 
but one of them was allowed to be in her own home at night..."
```

**After Segmentation**:
- Segment 1: "Three women were changed into flowers..."
- Segment 2: "She said to her husband, If thou wilt come..."
- Segment 3: "And he did so..."

**After Summarization**:
- Prompt 1: "Three women transformed into flowers in a field"
- Prompt 2: "A woman asks her husband to rescue her"
- Prompt 3: "A man saves his wife from flower form"

**After Image Generation**:
- `segment_0.jpg` - Image of three flowers in a field
- `segment_1.jpg` - Image of a woman and man talking
- `segment_2.jpg` - Image of a rescue scene

---

## ⚡ Performance

### **With GPU** (Your Setup):
- Segmentation: ~2-5 seconds per story
- Summarization: ~1-2 seconds per segment
- Image Generation: ~10-30 seconds per image
- **Total for 1 story (3 segments)**: ~1-2 minutes

### **With CPU** (Fallback):
- Segmentation: ~2-5 seconds per story
- Summarization: ~1-2 seconds per segment  
- Image Generation: ~5-10 minutes per image
- **Total for 1 story (3 segments)**: ~15-30 minutes

---

## 🔧 Current Status

✅ **Working**:
- Story loading
- GPU detection and usage
- Image generation (at least 1 image generated)
- Model loading

⚠️ **Needs Fix**:
- Segmentation (currently using fallback method)
- Some stories may need re-processing

---

## 🚀 Next Steps

1. **Re-run the script** to generate segments and summaries for all stories
2. **Generate images** for all stories (will use GPU for speed)
3. **View storyboards** in the `generated_images` folder

---

## 📊 Architecture Flow

```
Story Text
    ↓
[Segmentation] → Story Segments
    ↓
[Summarization] → Text Prompts
    ↓
[Image Generation] → Storyboard Images
    ↓
Organized Image Folders
```

---

## 💡 Key Features

1. **Automatic**: No manual intervention needed
2. **Consistent**: Images are conditioned on previous images for visual coherence
3. **Scalable**: Can process hundreds of stories
4. **Flexible**: Works with different story types
5. **GPU-Accelerated**: Fast generation with your NVIDIA GPU


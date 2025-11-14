import os
from utils import read_data
from segmentation import generateSegments, generateSummaries, summarize_with_pegasus
from image_generation import generate_images
# os.environ['CUDA_VISIBLE_DEVICES']='3'  # Commented out - will use CPU if no GPU available
import nltk
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')
from nltk.tokenize import TextTilingTokenizer

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

data_folder = os.path.join(project_root, "data", "clean_dataset")
images_out_dir = os.path.join(project_root, "generated_images", "baseline2_try")
os.makedirs(images_out_dir, exist_ok=True)
df = read_data(data_folder)

df = generateSegments(df)

summarizer_model_id = "sshleifer/distilbart-xsum-12-6"

df1 = generateSummaries(df, summarizer_model_id)

df1.to_csv(images_out_dir+"/df_summary_distilbart.csv")


df2 = summarize_with_pegasus(df)
df2.to_csv(images_out_dir+"/df_summary_pegasus.csv")

# Generate storyboard images from the summarized stories
# Note: This will download ~5GB Stable Diffusion model on first run
#       and may take a long time on CPU (several minutes per image)
print("\n" + "="*50)
print("Starting image generation (storyboard creation)...")
print("WARNING: This will take a LONG time on CPU!")
print("="*50 + "\n")
generate_images(df2, images_out_dir)
import nltk
import sys
import tqdm
import os
import torch
from nltk.tokenize import TextTilingTokenizer, sent_tokenize

# Download required NLTK resources (with error handling)
def download_nltk_resources():
    """Download required NLTK resources if not already present"""
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK resource '{resource}': {e}")
            # Try alternative download method
            try:
                nltk.download(resource, quiet=False)
            except:
                pass

# Download resources on import
download_nltk_resources()

from transformers import pipeline
from collections import defaultdict
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Only needed for debugging CUDA issues
ttt = TextTilingTokenizer(w = 10, k=5)
device = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect device
from transformers import PegasusForConditionalGeneration, PegasusTokenizer







def clean_segments(segments):
    clean_segment = []
    for segment in segments:
        if len(segment.strip())>10:
            clean_segment.append(segment.strip())
    return clean_segment


def generateSegments(df):
    print("Cleaning segments")
    works = 0
    not_works = 0
    all_segments = []

    for index, row in tqdm.tqdm(df.iterrows()):
    
        story = row["original_story"]
        
        # First, try to split by double newlines (paragraph breaks)
        paragraphs = [p.strip() for p in story.split('\n\n') if p.strip()]
        
        # If we have multiple paragraphs, use them as segments
        if len(paragraphs) > 1:
            print(f"  Found {len(paragraphs)} paragraphs, using as segments")
            # Clean and validate each paragraph
            valid_paragraphs = []
            for para in paragraphs:
                if len(para.strip()) > 20:  # Minimum length for a valid segment
                    # If paragraph is very long, split it into smaller chunks
                    sentences = sent_tokenize(para)
                    if len(sentences) > 8:
                        # Split long paragraphs into chunks of 4-6 sentences
                        for i in range(0, len(sentences), 5):
                            chunk = ' '.join(sentences[i:i+5])
                            if len(chunk.strip()) > 20:
                                valid_paragraphs.append(chunk.strip())
                    else:
                        valid_paragraphs.append(para.strip())
            
            if len(valid_paragraphs) > 0:
                all_segments.append(valid_paragraphs)
                works = works + 1
                continue
        
        # If no paragraph breaks, try TextTilingTokenizer
        try:
            # Try TextTilingTokenizer (works for longer texts with paragraph breaks)
            segments = ttt.tokenize(story)
            new_segments = []
            for segment in segments:
                sentences = sent_tokenize(segment)
                # Use smaller chunks (3-4 sentences) for better scene separation
                for i in range(0, len(sentences), 4):
                    new_segment = ' '.join(sentences[i:i+4])
                    if len(new_segment.strip()) > 20:
                        new_segments.append(new_segment.strip())
                    
            if len(new_segments) > 0:
                all_segments.append(clean_segments(new_segments))
                works = works+1
                continue
        except (ValueError, StopIteration) as e:
            pass  # Fall through to sentence-based splitting
            
        # Fallback: Split by sentences into chunks
        try:
            sentences = sent_tokenize(story)
            new_segments = []
            # Use smaller chunks (3-4 sentences) for better scene separation
            for i in range(0, len(sentences), 4):
                new_segment = ' '.join(sentences[i:i+4])
                if len(new_segment.strip()) > 20:
                    new_segments.append(new_segment.strip())
            
            if len(new_segments) > 1:  # Only use if we get multiple segments
                all_segments.append(new_segments)
                works = works + 1
            elif len(new_segments) == 1:
                # If only 1 segment but story is long, try splitting more aggressively
                if len(story) > 500:
                    # Split into smaller chunks of 2-3 sentences
                    new_segments = []
                    for i in range(0, len(sentences), 3):
                        new_segment = ' '.join(sentences[i:i+3])
                        if len(new_segment.strip()) > 20:
                            new_segments.append(new_segment.strip())
                    if len(new_segments) > 1:
                        all_segments.append(new_segments)
                        works = works + 1
                    else:
                        all_segments.append(new_segments)
                        works = works + 1
                else:
                    all_segments.append(new_segments)
                    works = works + 1
            else:
                # If still no segments, use the whole story as one segment
                if len(story.strip()) > 10:
                    all_segments.append([story.strip()])
                    works = works + 1
                else:
                    all_segments.append([])
                    not_works = not_works + 1
        except Exception as e2:
            # Last resort: use whole story as one segment
            if len(story.strip()) > 10:
                all_segments.append([story.strip()])
                works = works + 1
            else:
                all_segments.append([])
                not_works = not_works + 1
            
    print("segmentation works", works)
    print("segmentation doesn't work", not_works)
    
    df["segmented_story"] = all_segments
    return df


def generateSummaries(df, summarizer_model_id):
    
    pipe = pipeline("summarization", model=summarizer_model_id, max_length=40, device=0 if device == "cuda" else -1)
    all_summaries = []
    
    for index, row in tqdm.tqdm(df.iterrows()):
        summaries = []
        segments = row["segmented_story"]
        for segment in segments:
            try:
                if(len(segment.split(" "))>20):
                    summary = pipe(segment)
                    summaries.append(summary[0]['summary_text'])
                else:
                    summaries.append(segment)
            except Exception as e:
                print(f"An error occurred with model {summarizer_model_id}: {e}")
                summaries.append(segment)
                #append original segment in case of error
        all_summaries.append(summaries)
        
    df["summarized_story"] = all_summaries
    return df


def summarize_with_pegasus(df, model_name="google/pegasus-xsum", max_length=40, min_length=10):
    
    all_summaries = []
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    for index, row in tqdm.tqdm(df.iterrows()):
        summaries = []
        segments = row["segmented_story"]
        for segment in segments:
            try:
                if(len(segment.split(" "))>20):
                    tokens = tokenizer(segment, truncation=True, padding="longest", return_tensors="pt").to(device)
                    summary_ids = model.generate(tokens["input_ids"], max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
                    summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
                else:
                    summaries.append(segment)
            except Exception as e:
                print(f"An error occurred with model {model_name}: {e}")
                summaries.append(segment)
                #append original segment in case of error
        all_summaries.append(summaries)
        
    df["summarized_story"] = all_summaries
    return df

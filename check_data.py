import pandas as pd
import os

# Check CSV data
df = pd.read_csv('generated_images/baseline2_try/df_summary_pegasus.csv')
print("="*60)
print("DATA CHECK FOR IMAGE GENERATION")
print("="*60)
print(f"\nTotal stories in CSV: {len(df)}")

# Check original stories
stories_with_content = df[df['original_story'].astype(str).str.len() > 100]
print(f"Stories with content (>100 chars): {len(stories_with_content)}/{len(df)}")

# Check segments
empty_segments = df[df['segmented_story'].astype(str) == '[]']
print(f"Stories with EMPTY segments: {len(empty_segments)}/{len(df)}")

# Check summaries  
empty_summaries = df[df['summarized_story'].astype(str) == '[]']
print(f"Stories with EMPTY summaries: {len(empty_summaries)}/{len(df)}")

# Sample story
print("\n" + "="*60)
print("SAMPLE STORY:")
print("="*60)
sample = df.iloc[0]
print(f"Name: {sample['story_name']}")
print(f"Type: {sample['story_type']}")
print(f"Original story length: {len(str(sample['original_story']))} chars")
print(f"Original story preview: {str(sample['original_story'])[:200]}...")
print(f"Has segments: {sample['segmented_story'] != '[]'}")
print(f"Has summaries: {sample['summarized_story'] != '[]'}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
if len(empty_segments) == len(df):
    print("❌ NO SEGMENTS - Need to re-run segmentation with fixed code")
    print("✅ Original stories exist - Can generate images from them")
    print("\nACTION: Re-run main.py to generate segments and summaries")
else:
    print("✅ Segments exist - Ready for image generation!")
    print("✅ Summaries exist - Ready for image generation!")


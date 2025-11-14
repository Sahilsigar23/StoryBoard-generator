import torch
from PIL import Image
from io import BytesIO
import os
import numpy as np
import warnings
import logging
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionImg2ImgPipeline
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in cast.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Couldn\'t connect to the Hub.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Repository Not Found.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Invalid username or password.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Keyword arguments.*are not expected.*')

# Suppress Hugging Face Hub connection warnings and errors
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("diffusers").setLevel(logging.ERROR)

# Suppress urllib3 connection warnings
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Auto-detect device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize models (lazy loading - only when generate_images is called)
model_id = "stabilityai/stable-diffusion-2"
text2img_pipe = None
img2img_pipe = None
bert_tokenizer = None
bert_model = None

def _initialize_models():
    """Initialize models only when needed"""
    global text2img_pipe, img2img_pipe, bert_tokenizer, bert_model
    
    if text2img_pipe is None:
        print("Loading Stable Diffusion models (this may take a while and download ~5GB)...")
        print("Note: If you see connection errors, models will load from local cache...")
        try:
            scheduler = EulerDiscreteScheduler.from_pretrained(
                model_id, 
                subfolder="scheduler"
            )
            # Use float32 for stability (float16 causes NaN issues with this model)
            # Use sequential CPU offloading to manage memory
            dtype = torch.float32
            
            # Clear GPU cache before loading
            if device == "cuda":
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Enable attention slicing to reduce memory usage
            text2img_pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                scheduler=scheduler, 
                torch_dtype=dtype,
                safety_checker=None,  # Disable safety checker to avoid issues
                requires_safety_checker=False
            )
            
            # Enable sequential CPU offloading to save GPU memory (moves models to CPU when not in use)
            if device == "cuda" and hasattr(text2img_pipe, "enable_sequential_cpu_offload"):
                text2img_pipe.enable_sequential_cpu_offload()
            elif device == "cuda" and hasattr(text2img_pipe, "enable_model_cpu_offload"):
                text2img_pipe.enable_model_cpu_offload()
            else:
                text2img_pipe = text2img_pipe.to(device)
            
            # Enable memory efficient attention
            if hasattr(text2img_pipe, "enable_attention_slicing"):
                text2img_pipe.enable_attention_slicing()
            # Enable VAE slicing for memory efficiency
            if hasattr(text2img_pipe, "enable_vae_slicing"):
                text2img_pipe.enable_vae_slicing()
            
            img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Enable sequential CPU offloading for img2img too
            if device == "cuda" and hasattr(img2img_pipe, "enable_sequential_cpu_offload"):
                img2img_pipe.enable_sequential_cpu_offload()
            elif device == "cuda" and hasattr(img2img_pipe, "enable_model_cpu_offload"):
                img2img_pipe.enable_model_cpu_offload()
            else:
                img2img_pipe = img2img_pipe.to(device)
            
            # Enable memory efficient attention for img2img
            if hasattr(img2img_pipe, "enable_attention_slicing"):
                img2img_pipe.enable_attention_slicing()
            # Enable VAE slicing for memory efficiency
            if hasattr(img2img_pipe, "enable_vae_slicing"):
                img2img_pipe.enable_vae_slicing()
            
            print("[OK] Stable Diffusion models loaded!")
        except Exception as e:
            print(f"[ERROR] Error loading Stable Diffusion models: {e}")
            raise
    
    if bert_tokenizer is None:
        print("Loading BERT model...")
        try:
            bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
            print("[OK] BERT model loaded!")
        except Exception as e:
            print(f"[ERROR] Error loading BERT model: {e}")
            raise


# prefix = "Generate an artistic interpretation of the text "
# prefix = ""

def enhance_prompt(prompt):
    """
    Enhance the prompt to make Stable Diffusion follow it better.
    Adds descriptive keywords and improves clarity to ensure the model
    follows the specific details in the prompt.
    """
    if not prompt or len(prompt.strip()) == 0:
        return prompt
    
    prompt = prompt.strip()
    
    # Check if prompt already has quality keywords
    has_quality = any(word in prompt.lower() for word in ['high quality', 'detailed', 'sharp', 'professional'])
    
    # Add quality and style keywords to improve adherence
    quality_keywords = "high quality, detailed, sharp focus, professional photography, accurate depiction"
    
    # Make the prompt more explicit - ensure key details are emphasized
    # If the prompt mentions specific actions or objects, make sure they're clear
    enhanced = prompt
    
    # Add emphasis to important elements
    # Check for key action words that should be emphasized
    action_words = ['smiling', 'drinking', 'standing', 'sitting', 'walking', 'looking', 'holding']
    has_action = any(word in prompt.lower() for word in action_words)
    
    # If prompt has specific details, ensure they're preserved
    if has_action or len(prompt.split(',')) >= 2:
        # Prompt already has details, just add quality keywords
        enhanced = f"{prompt}, {quality_keywords}"
    else:
        # Short prompt, add more detail and quality
        enhanced = f"{prompt}, {quality_keywords}"
    
    # Ensure the prompt is clear about what to show
    # Remove any ambiguity by making descriptions more explicit
    return enhanced

def generate_images(df, out_dir):
    """Generate images from summarized story prompts"""
    _initialize_models()  # Load models when needed
    
    for index, row in df.iterrows():
        prompts = row["summarized_story"]
        story_type = row["story_type"]
        story_name = row["story_name"]
        original_story = row.get("original_story", "")
        
        # Handle case where prompts might be stored as string representation of list
        if isinstance(prompts, str):
            try:
                import ast
                prompts = ast.literal_eval(prompts)
            except:
                prompts = []
        
        # If no prompts (segmentation failed), create a simple summary from original story
        if not prompts or len(prompts) == 0:
            if original_story and len(original_story.strip()) > 0:
                # Create a simple prompt from the first few sentences
                sentences = original_story.split('.')[:3]  # First 3 sentences
                simple_prompt = '. '.join(sentences).strip()
                if len(simple_prompt) > 20:
                    prompts = [simple_prompt[:200]]  # Limit length
                    print(f"Using simple summary for {story_name} (segmentation failed)")
                else:
                    print(f"Skipping {story_name} - story too short")
                    continue
            else:
                print(f"Skipping {story_name} - no prompts or story available")
                continue
            
        print(f"Generating images for {story_name} ({len(prompts)} segments)...")
        print(f"  Prompts: {[str(p)[:50] + '...' if len(str(p)) > 50 else str(p) for p in prompts]}")
        
        # Create story-specific directory
        # Check if out_dir already ends with story_type/story_name to avoid nesting
        out_dir_normalized = os.path.normpath(out_dir)
        expected_suffix = os.path.normpath(os.path.join(story_type, story_name))
        
        if out_dir_normalized.endswith(expected_suffix):
            # out_dir already contains the full path, use it directly
            story_dir = out_dir
        else:
            # Need to add story_type and story_name
            story_dir = os.path.join(out_dir, story_type, story_name)
        
        os.makedirs(story_dir, exist_ok=True)
        print(f"  Output directory: {story_dir}")
        
        # Verify we have multiple prompts
        if len(prompts) == 1:
            print(f"  [WARNING] Only 1 prompt found. Expected multiple segments for storyboard.")
            print(f"  [INFO] This might be because segmentation created only 1 segment.")
        
        try:
            # Generate first image from first prompt
            init_prompt = prompts[0] if isinstance(prompts, list) else str(prompts)
            
            # Enhance prompt for better image generation
            # Add descriptive keywords to make the model follow the prompt better
            enhanced_prompt = enhance_prompt(init_prompt)
            print(f"  Generating segment 0: {init_prompt[:50]}...")
            print(f"  Enhanced prompt: {enhanced_prompt[:80]}...")
            
            # Generate with error handling
            try:
                import numpy as np
                import random
                
                # Clear GPU cache before generation
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                with torch.no_grad():
                    # Use a random seed for variety
                    generator = torch.Generator(device=device).manual_seed(random.randint(0, 2**32))
                    
                    # Generate image with enhanced prompt and better parameters
                    # Use more steps and higher guidance for better prompt adherence
                    result = text2img_pipe(
                        enhanced_prompt,  # Use enhanced prompt
                        num_inference_steps=50,  # More steps for better quality
                        guidance_scale=9.0,  # Higher guidance to follow prompt better
                        height=512,
                        width=512,
                        generator=generator
                    )
                    
                    init_image = result.images[0]
                    
                    # Post-process to fix any NaN/Inf issues from VAE
                    img_array = np.array(init_image)
                    
                    # If image is all black or has NaN/Inf, fix it
                    if img_array.mean() < 5.0 or np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                        print(f"  [WARNING] Image has issues (mean: {img_array.mean():.2f}), fixing...")
                        # Fix NaN/Inf
                        if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                            # Use median of valid pixels
                            valid_mask = ~(np.isnan(img_array) | np.isinf(img_array))
                            if np.any(valid_mask):
                                median_val = np.median(img_array[valid_mask])
                                img_array = np.where(np.isnan(img_array) | np.isinf(img_array), median_val, img_array)
                            else:
                                # All invalid - use gray
                                img_array = np.full_like(img_array, 128, dtype=np.uint8)
                        
                        # If still all black, regenerate with different seed
                        if img_array.mean() < 5.0:
                            print(f"  [WARNING] Regenerating with different seed...")
                            generator2 = torch.Generator(device=device).manual_seed(random.randint(0, 2**32))
                            if device == "cuda":
                                torch.cuda.empty_cache()
                            with torch.no_grad():
                                result2 = text2img_pipe(
                                    enhanced_prompt,  # Use enhanced prompt
                                    num_inference_steps=50,  # More steps for better quality
                                    guidance_scale=9.0,  # Higher guidance
                                    height=512,
                                    width=512,
                                    generator=generator2
                                )
                                init_image = result2.images[0]
                                img_array = np.array(init_image)
                        
                        # Final fix if needed
                        if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                            img_array = np.nan_to_num(img_array, nan=128, posinf=255, neginf=0)
                        
                        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                        init_image = Image.fromarray(img_array)
                
                # Validate image before saving
                if init_image is None:
                    raise ValueError("Generated image is None")
                
                # Check if image is valid (not all black)
                import numpy as np
                img_array = np.array(init_image)
                
                # If image is all black or has very low mean, there's a problem
                if img_array.mean() < 5.0 or np.all(img_array < 10):
                    print(f"  [WARNING] Generated image is all black, trying different approach...")
                    # Try with different seed
                    import random
                    generator = torch.Generator(device=device).manual_seed(random.randint(0, 2**32))
                    result = text2img_pipe(
                        init_prompt,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        height=512,
                        width=512,
                        generator=generator
                    )
                    init_image = result.images[0]
                    img_array = np.array(init_image)
                    
                    # If still black, there's a deeper issue
                    if img_array.mean() < 5.0:
                        raise ValueError(f"Image generation failed - produced black image. Mean value: {img_array.mean()}")
                
                # Ensure image is in RGB mode
                if init_image.mode != 'RGB':
                    init_image = init_image.convert('RGB')
                
                # Final validation - check for NaN/Inf and fix if needed
                img_array = np.array(init_image)
                if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                    print(f"  [WARNING] Image contains NaN/Inf values, fixing...")
                    # Use a better approach - replace with nearby valid pixels
                    valid_mask = ~(np.isnan(img_array) | np.isinf(img_array))
                    if np.any(valid_mask):
                        # Fill NaN/Inf with median of valid pixels
                        median_val = np.median(img_array[valid_mask])
                        img_array = np.where(np.isnan(img_array) | np.isinf(img_array), median_val, img_array)
                    else:
                        # If all invalid, use a default gray
                        img_array = np.full_like(img_array, 128, dtype=np.uint8)
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    init_image = Image.fromarray(img_array)
                
                save_path = os.path.join(story_dir, f"segment_0.jpg")
                init_image.save(save_path, quality=95, format='JPEG')
                print(f"  [OK] Saved segment_0.jpg (mean pixel value: {np.array(init_image).mean():.1f})")
                
            except Exception as img_error:
                print(f"  [ERROR] Error generating first image: {img_error}")
                import traceback
                traceback.print_exc()
                continue
            
            # Generate subsequent images conditioned on the first
            num_additional_segments = len(prompts) - 1
            print(f"  Generating {num_additional_segments} additional segment(s)...")
            
            if num_additional_segments == 0:
                print(f"  [INFO] Only 1 segment found. Only segment_0.jpg will be generated.")
            
            for idx, prompt in enumerate(prompts[1:]):
                prompt_str = prompt if isinstance(prompt, str) else str(prompt)
                
                # Enhance prompt for better adherence
                enhanced_prompt_str = enhance_prompt(prompt_str)
                
                segment_num = idx + 1
                print(f"  Generating segment {segment_num}/{num_additional_segments}: {prompt_str[:50]}...")
                print(f"  Enhanced prompt: {enhanced_prompt_str[:80]}...")
                
                try:
                    with torch.no_grad():
                        result = img2img_pipe(
                            prompt=enhanced_prompt_str,  # Use enhanced prompt
                            image=init_image,
                            strength=0.75,  # Reduced from 0.90 for better stability
                            guidance_scale=9.0,  # Higher guidance to follow prompt better
                            num_inference_steps=50  # More steps for better quality
                        )
                    generated_image = result.images[0]
                    
                    # Validate image
                    if generated_image is None:
                        raise ValueError(f"Generated image {idx+1} is None")
                    
                    # Check if image is valid (not all black)
                    import numpy as np
                    img_array = np.array(generated_image)
                    
                    # If image is all black, try regenerating with different seed
                    if img_array.mean() < 5.0 or np.all(img_array < 10):
                        print(f"  [WARNING] Image {idx+1} is all black, trying different seed...")
                        import random
                        generator = torch.Generator(device=device).manual_seed(random.randint(0, 2**32))
                        with torch.no_grad():
                            result = img2img_pipe(
                                prompt=enhanced_prompt_str,  # Use enhanced prompt
                                image=init_image,
                                strength=0.75,
                                guidance_scale=9.0,  # Higher guidance
                                num_inference_steps=50,
                                generator=generator
                            )
                        generated_image = result.images[0]
                        img_array = np.array(generated_image)
                        
                        if img_array.mean() < 5.0:
                            raise ValueError(f"Image {idx+1} generation failed - produced black image")
                    
                    # Ensure image is in RGB mode
                    if generated_image.mode != 'RGB':
                        generated_image = generated_image.convert('RGB')
                    
                    # Final validation - check for NaN/Inf
                    img_array = np.array(generated_image)
                    if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                        print(f"  [WARNING] Image {idx+1} contains NaN/Inf values, fixing...")
                        valid_mask = ~(np.isnan(img_array) | np.isinf(img_array))
                        if np.any(valid_mask):
                            median_val = np.median(img_array[valid_mask])
                            img_array = np.where(np.isnan(img_array) | np.isinf(img_array), median_val, img_array)
                        else:
                            img_array = np.full_like(img_array, 128, dtype=np.uint8)
                        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                        generated_image = Image.fromarray(img_array)
                    
                    save_path = os.path.join(story_dir, f"segment_{idx+1}.jpg")
                    generated_image.save(save_path, quality=95, format='JPEG')
                    print(f"  [OK] Saved segment_{idx+1}.jpg (mean pixel value: {np.array(generated_image).mean():.1f})")
                    
                    init_image = generated_image  # Use generated image for next iteration
                    
                except Exception as img_error:
                    print(f"  [ERROR] Error generating segment {idx+1}: {img_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue with next segment instead of stopping
                    continue
                
        except Exception as e:
            print(f"  [ERROR] Error generating images for {story_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


def generate_images_baseline0(prompts, prefix=""):
    _initialize_models()  # Ensure models are loaded
    generated_images = []
    if len(prompts)>0:
        for idx, prompt in enumerate(prompts):    
            image = text2img_pipe(prefix+prompt).images[0]            
            generated_images.append(image)
    return generated_images


def generate_images_baseline(prompts, strength, guidance_scale, prefix=""):
    _initialize_models()  # Ensure models are loaded
    if len(prompts)>0:
        init_prompt = prompts[0]
        init_image = text2img_pipe(prefix+init_prompt).images[0]
        generated_images = [init_image]
        for idx, prompt in enumerate(prompts[1:]):    
            images = img2img_pipe(prompt=prefix+prompt, image=init_image, strength=strength, guidance_scale=guidance_scale).images
            generated_images.append(images[0])
    return generated_images

def bert_sentence_embedding(sentence):
    _initialize_models()  # Ensure models are loaded
    tokens = bert_tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        output = bert_model(**tokens.to(device))
    # Extract the embeddings for the [CLS] token
    cls_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embedding

def findSimilarPromptBert(query, string_list):
    if not string_list:
        return None
    query_embedding = bert_sentence_embedding(query)
    string_embeddings = [bert_sentence_embedding(s) for s in string_list]

    similarities = [cosine_similarity(query_embedding, s.reshape(1, -1))[0, 0] for s in string_embeddings]
    most_similar_index = similarities.index(max(similarities))
    return most_similar_index


def generate_images_proposed(story, strength, guidance_scale, prefix=""):
    _initialize_models()  # Ensure models are loaded
    init_prompt = story[0]
    # print(init_prompt)
    init_image = text2img_pipe(prefix+init_prompt).images[0]
    generated_images = [init_image]
    for idx, prompt in enumerate(story[1:]):
        most_similar_index = findSimilarPromptBert(prompt, story[:idx+1])
        print("CURRENT PROMPT:",prompt)
        print("SIMILAR PROMPT", story[most_similar_index])
        images = img2img_pipe(prompt=prefix+prompt, image=generated_images[most_similar_index], strength=strength, guidance_scale=guidance_scale).images
        generated_images.append(images[0])
    return generated_images


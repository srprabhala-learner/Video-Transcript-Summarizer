import streamlit as st
from dotenv import load_dotenv
import sys

load_dotenv() ##load all the environment variables
import os

# Enable console logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_print(message):
    """Print to console and show in Streamlit if callback available"""
    print(f"🔍 DEBUG: {message}", file=sys.stderr)
    logger.debug(message)

# Use HuggingFace Inference API (faster) or Ollama (local) as fallback
try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    InferenceClient = None
    HUGGINGFACE_AVAILABLE = False

# Ollama as fallback
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    ChatOllama = None
    OLLAMA_AVAILABLE = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YouTubeTranscriptApi = None
    YOUTUBE_API_AVAILABLE = False

def get_prompt(language_name=None):
    """Generate prompt based on transcript language"""
    base_prompt = """You are a helpful assistant that summarizes YouTube videos. Given a video transcript, provide a concise summary of the content. If the video is a movie, describe the key scenes and plot points. If it's educational, highlight the main points and educational value. Return your response in markdown format not more than 1000 words."""
    
    if language_name and language_name != 'English':
        # Enhanced prompt for non-English languages with explicit language instruction
        prompt = f"""{base_prompt}

IMPORTANT: The transcript is in {language_name}. 
- Read and understand the transcript in {language_name}
- Provide your summary EXCLUSIVELY in {language_name}
- Do NOT translate to English
- Maintain the original language throughout your response
- Use proper grammar and vocabulary in {language_name}

Transcript:
"""
    else:
        prompt = f"""{base_prompt}

Please provide the summary of the text given here: """
    
    return prompt


## getting the transcript data from yt videos
def extract_transcript_details(youtube_video_url, preferred_language='en'):
    try:
        # Extract video ID from different YouTube URL formats
        if "youtube.com/watch?v=" in youtube_video_url:
            video_id = youtube_video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_video_url:
            video_id = youtube_video_url.split("youtu.be/")[1].split("?")[0]
        elif "youtube.com/embed/" in youtube_video_url:
            video_id = youtube_video_url.split("embed/")[1].split("?")[0]
        else:
            # Fallback: try to extract after last '='
            video_id = youtube_video_url.split("=")[-1].split("&")[0]
        
        # Get transcript - handle both old and new API versions
        if not YOUTUBE_API_AVAILABLE or YouTubeTranscriptApi is None:
            raise ImportError("youtube-transcript-api package not installed. Run: pip install youtube-transcript-api")
        
        transcript_list = None
        used_language = None
        
        # Strategy: Try new API first (v1.0.0+), then fallback to old API
        # Check if we can instantiate (new API) or use class methods (old API)
        can_instantiate = False
        try:
            test_api = YouTubeTranscriptApi()
            can_instantiate = True
        except (TypeError, AttributeError):
            can_instantiate = False
        
        if can_instantiate:
            # NEW API (v1.0.0+): Create instance and use fetch()
            ytt_api = YouTubeTranscriptApi()
            available_languages = []
            
            # Try to get list of available transcripts first
            # In new API, list_transcripts might be a class method
            try:
                # Try as class method first (most common in new API)
                transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # Get all available languages - iterate through the transcript list
                for transcript in transcript_list_obj:
                    available_languages.append({
                        'code': transcript.language_code,
                        'name': transcript.language,
                        'generated': transcript.is_generated
                    })
            except (AttributeError, TypeError) as attr_err:
                # If class method doesn't work, list_transcripts might not be available
                # We'll try fetching without language parameter (gets default/first available)
                pass
            except Exception as list_err:
                # If list_transcripts fails with other error, we'll try fetch without language
                # The error message might contain available languages, but we'll handle it in fetch
                pass
            
            # Try preferred language first (English) if available, otherwise use first available
            transcript_found = False
            
            if available_languages:
                # Check if English is in the available languages - CRITICAL: only try English if it exists
                english_available = any(lang['code'] == preferred_language for lang in available_languages)
                
                if english_available:
                    # Try English first ONLY if it's in the available languages list
                    try:
                        transcript_data = ytt_api.fetch(video_id, languages=[preferred_language])
                        transcript_list = transcript_data.to_raw_data()
                        used_language = next(lang for lang in available_languages if lang['code'] == preferred_language)
                        transcript_found = True
                    except Exception as eng_err:
                        # If English fetch fails, mark as not found and use fallback
                        transcript_found = False
                        # Don't raise - we'll try the first available language instead
                
                # If English not available or failed, use first available language
                # IMPORTANT: This should ALWAYS execute if English is not available
                if not transcript_found and available_languages:
                    first_lang = available_languages[0]
                    try:
                        # Try fetching with the first available language code (e.g., Telugu)
                        transcript_data = ytt_api.fetch(video_id, languages=[first_lang['code']])
                        transcript_list = transcript_data.to_raw_data()
                        used_language = first_lang
                        if first_lang['code'] != preferred_language:
                            st.info(f"English transcript not available. Using {first_lang['name']} ({first_lang['code']}) transcript instead.")
                        transcript_found = True
                    except Exception as fetch_error:
                        # If fetch with language code fails, try without language parameter
                        try:
                            transcript_data = ytt_api.fetch(video_id)
                            transcript_list = transcript_data.to_raw_data()
                            used_language = first_lang
                            if first_lang['code'] != preferred_language:
                                st.info(f"Using {first_lang['name']} ({first_lang['code']}) transcript.")
                            transcript_found = True
                        except Exception as final_err:
                            # Last resort: raise with helpful message
                            available_codes = [lang['code'] for lang in available_languages]
                            raise Exception(f"Could not fetch transcript. Available languages: {available_codes}. Error: {str(final_err)}")
                
                if not transcript_found and available_languages:
                    # Final attempt: use first available language
                    first_lang = available_languages[0]
                    transcript_data = ytt_api.fetch(video_id, languages=[first_lang['code']])
                    transcript_list = transcript_data.to_raw_data()
                    used_language = first_lang
                    if first_lang['code'] != preferred_language:
                        st.info(f"Using {first_lang['name']} ({first_lang['code']}) transcript.")
                elif not transcript_found:
                    raise Exception(f"Failed to fetch transcript. Available languages: {[lang['code'] for lang in available_languages] if available_languages else 'unknown'}")
            else:
                # If we couldn't get language list, the error message will tell us what's available
                # IMPORTANT: Never try English specifically - parse error to get available languages
                try:
                    # First, try to get list one more time using class method
                    try:
                        transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
                        for transcript in transcript_list_obj:
                            available_languages.append({
                                'code': transcript.language_code,
                                'name': transcript.language,
                                'generated': transcript.is_generated
                            })
                        
                        # If we got languages, use first available
                        if available_languages:
                            first_lang = available_languages[0]
                            transcript_data = ytt_api.fetch(video_id, languages=[first_lang['code']])
                            transcript_list = transcript_data.to_raw_data()
                            used_language = first_lang
                            if first_lang['code'] != preferred_language:
                                st.info(f"English transcript not available. Using {first_lang['name']} ({first_lang['code']}) transcript instead.")
                        else:
                            raise Exception("No languages found in list")
                    except Exception:
                        # If list_transcripts still fails, we need to trigger an error to get available languages
                        # Try fetching with English to get the error message with available languages
                        try:
                            transcript_data = ytt_api.fetch(video_id, languages=[preferred_language])
                            transcript_list = transcript_data.to_raw_data()
                            used_language = {'code': preferred_language, 'name': 'English'}
                        except Exception as eng_error:
                            # This error will contain available languages - parse it
                            error_str = str(eng_error)
                            
                            # Parse available languages from error message
                            # Format examples:
                            # "te (\"Telugu (auto-generated)\")"
                            # "te (Telugu (auto-generated))"
                            import re
                            
                            # Try to match language codes with quoted names
                            lang_pattern = r'(\w+)\s*\([^)]*"([^"]+)"[^)]*\)'
                            lang_matches = re.findall(lang_pattern, error_str)
                            
                            if not lang_matches:
                                # Try pattern without quotes: "te (Telugu (auto-generated))"
                                lang_pattern = r'(\w+)\s*\(([^)]+)\)'
                                lang_matches = re.findall(lang_pattern, error_str)
                            
                            if not lang_matches:
                                # Try simplest pattern: just 2-letter language codes before parentheses
                                lang_pattern = r'\b([a-z]{2})\s*\('
                                lang_codes = re.findall(lang_pattern, error_str)
                                lang_matches = [(code, code.upper()) for code in lang_codes]
                            
                            # Filter out 'en' if it's in the matches (since we know it's not available)
                            lang_matches = [m for m in lang_matches if m[0] != 'en']
                            
                            if lang_matches:
                                # Use first available language from error
                                first_lang_code = lang_matches[0][0]
                                first_lang_name = lang_matches[0][1] if lang_matches[0][1] else first_lang_code
                                
                                try:
                                    transcript_data = ytt_api.fetch(video_id, languages=[first_lang_code])
                                    transcript_list = transcript_data.to_raw_data()
                                    used_language = {'code': first_lang_code, 'name': first_lang_name}
                                    st.info(f"English transcript not available. Using {first_lang_name} ({first_lang_code}) transcript instead.")
                                except Exception as fetch_err:
                                    raise Exception(f"Could not fetch transcript in {first_lang_code}. Error: {str(fetch_err)}")
                            else:
                                # If we can't parse, raise original error
                                raise eng_error
                except Exception as final_error:
                    raise final_error
        else:
            # OLD API: Use class methods
            try:
                # Try to get list of available transcripts
                try:
                    transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
                    available_languages = []
                    
                    for transcript in transcript_list_obj:
                        available_languages.append({
                            'code': transcript.language_code,
                            'name': transcript.language,
                            'generated': transcript.is_generated
                        })
                    
                    # Try preferred language first
                    transcript_found = False
                    for lang_info in available_languages:
                        if lang_info['code'] == preferred_language:
                            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[preferred_language])
                            used_language = lang_info
                            transcript_found = True
                            break
                    
                    # If preferred not found, use first available
                    if not transcript_found and available_languages:
                        first_lang = available_languages[0]
                        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[first_lang['code']])
                        used_language = first_lang
                        st.info(f"English transcript not available. Using {first_lang['name']} ({first_lang['code']}) transcript instead.")
                        
                except AttributeError:
                    # Last resort: try get_transcript without language
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    used_language = {'code': 'auto', 'name': 'Auto-detected'}
                    
            except AttributeError as e:
                raise Exception(f"youtube-transcript-api version incompatible. Error: {str(e)}. Please upgrade: pip install --upgrade youtube-transcript-api")
            except Exception as e:
                raise e

        if transcript_list is None:
            raise Exception("Could not retrieve transcript")
        
        # Combine all transcript text
        transcript = ""
        for item in transcript_list:
            transcript += " " + item["text"]

        return transcript.strip(), used_language

    except Exception as e:
        st.error(f"Error extracting transcript: {str(e)}")
        st.info("Note: Some videos may not have transcripts available, or they may be disabled.")
        return None, None
    
def chunk_text(text, chunk_size=2500, overlap=300):
    """
    Split text into chunks with overlap to preserve context.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk (in characters)
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If not the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            for punct in ['. ', '.\n', '! ', '?\n', '? ']:
                last_punct = text.rfind(punct, start, end)
                if last_punct != -1 and last_punct > start + chunk_size // 2:
                    end = last_punct + len(punct)
                    break
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

## getting the summary based on Prompt from HuggingFace (cloud) or Ollama (local)
def generate_summary_content(transcript_text, prompt, progress_callback=None, use_local_model=False, _is_chunk=False):
    try:
        # DEBUG: Log entry point
        transcript_length = len(transcript_text) if transcript_text else 0
        debug_print(f"Starting generate_summary_content. use_local_model={use_local_model}, transcript_length={transcript_length}")
        if progress_callback:
            progress_callback(f"🔍 DEBUG: Starting generate_summary_content. use_local_model={use_local_model}, transcript_length={transcript_length}")
        
        # Determine if we need to chunk the transcript
        CHUNK_THRESHOLD = 3000  # Chunk if transcript is longer than this
        CHUNK_SIZE = 2500  # Size of each chunk
        CHUNK_OVERLAP = 300  # Overlap between chunks to preserve context
        
        should_chunk = not _is_chunk and transcript_length > CHUNK_THRESHOLD
        
        if should_chunk:
            debug_print(f"Transcript is long ({transcript_length} chars), will chunk into pieces")
            if progress_callback:
                progress_callback(f"📝 Long transcript detected ({transcript_length} chars). Chunking into pieces for better summarization...")
            
            # Chunk the transcript
            chunks = chunk_text(transcript_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            debug_print(f"Created {len(chunks)} chunks")
            if progress_callback:
                progress_callback(f"📝 Created {len(chunks)} chunks (avg {transcript_length // len(chunks)} chars each)")
            
            # Summarize each chunk
            chunk_summaries = []
            for idx, chunk in enumerate(chunks):
                debug_print(f"Summarizing chunk {idx+1}/{len(chunks)}")
                if progress_callback:
                    progress_callback(f"📝 Summarizing chunk {idx+1}/{len(chunks)} ({len(chunk)} chars)...")
                
                # Recursively call this function for each chunk (but don't chunk again)
                chunk_summary = generate_summary_content(
                    chunk, 
                    prompt, 
                    progress_callback=None,  # Don't show nested progress
                    use_local_model=use_local_model,
                    _is_chunk=True  # Prevent recursive chunking
                )
                
                if chunk_summary:
                    chunk_summaries.append(chunk_summary)
                    debug_print(f"Chunk {idx+1} summarized: {len(chunk_summary)} chars")
                else:
                    debug_print(f"Chunk {idx+1} failed to summarize")
                    # Add a placeholder if chunk summary fails
                    chunk_summaries.append(f"[Chunk {idx+1} summary unavailable]")
            
            # Combine chunk summaries
            if chunk_summaries:
                combined_summaries = "\n\n".join([f"**Part {i+1}:**\n{summary}" for i, summary in enumerate(chunk_summaries)])
                debug_print(f"Combined {len(chunk_summaries)} chunk summaries, total length: {len(combined_summaries)}")
                
                if progress_callback:
                    progress_callback(f"✅ Combined {len(chunk_summaries)} chunk summaries")
                
                # Optionally, create a final summary of the combined summaries if it's still long
                if len(combined_summaries) > 5000:
                    if progress_callback:
                        progress_callback("📝 Creating final summary of combined summaries...")
                    final_summary = generate_summary_content(
                        combined_summaries,
                        prompt,
                        progress_callback=None,
                        use_local_model=use_local_model,
                        _is_chunk=True  # Prevent further chunking
                    )
                    return final_summary if final_summary else combined_summaries
                else:
                    return combined_summaries
            else:
                debug_print("All chunk summaries failed")
                if progress_callback:
                    progress_callback("❌ All chunk summaries failed")
                return None
        
        # Use HuggingFace Inference API if selected (faster, cloud-based)
        if not use_local_model:
            if not HUGGINGFACE_AVAILABLE or InferenceClient is None:
                st.error("HuggingFace not available. Please install: pip install huggingface_hub")
                st.info("Falling back to Ollama...")
                use_local_model = True
            else:
                # HuggingFace is available, use it
                if progress_callback:
                    progress_callback("Using HuggingFace cloud model (fast)...")
                
                # Get API token from environment (optional, but recommended for rate limits)
                api_token = os.getenv("HUGGINGFACE_API_TOKEN")
                if progress_callback:
                    if api_token:
                        progress_callback(f"🔍 DEBUG: HUGGINGFACE_API_TOKEN found (length: {len(api_token)})")
                    else:
                        progress_callback("🔍 DEBUG: No HUGGINGFACE_API_TOKEN found (using public API, rate-limited)")
                
                # Use models that work well for summarization
                # Note: Some models work better with instruction prompts, others with simple prompts
                models_to_try = [
                    "mistralai/Mistral-Small-24B-Instruct-2501",                  # Very reliable, works with task prompts
             # Reliable, works with task prompts

                ]
                
                # Format prompt for HuggingFace models (some need instruction format)
                # For T5 models, use a simpler prompt format
                if "flan-t5" in models_to_try[0] or "flan-t5" in models_to_try[1]:
                    # T5 models work better with task-based prompts
                    full_prompt = f"Summarize the following transcript in detail:\n\n{transcript_text}"
                else:
                    # For instruction-tuned models, use the full prompt
                    full_prompt = prompt + "\n\nTranscript:\n" + transcript_text
                
                # Truncate if too long (HuggingFace has token limits)
                MAX_LENGTH = 2000
                if len(full_prompt) > MAX_LENGTH:
                    if progress_callback:
                        progress_callback(f"Truncating prompt ({len(full_prompt)} chars) to {MAX_LENGTH} chars...")
                    # Keep prompt and truncate transcript
                    if "flan-t5" in models_to_try[0] or "flan-t5" in models_to_try[1]:
                        transcript_part = transcript_text[:MAX_LENGTH - 50]
                        full_prompt = f"Summarize the following transcript in detail:\n\n{transcript_part}... [truncated]"
                    else:
                        prompt_len = len(prompt)
                        transcript_part = transcript_text[:MAX_LENGTH - prompt_len - 100]
                        full_prompt = prompt + "\n\nTranscript:\n" + transcript_part + "... [truncated]"
                
                # Initialize client with error handling
                try:
                    if progress_callback:
                        progress_callback("🔍 DEBUG: Initializing HuggingFace InferenceClient...")
                    
                    # Test if we can create a client (even without token, it should work for public models)
                    if api_token:
                        client = InferenceClient(token=api_token)
                        if progress_callback:
                            progress_callback("✅ HuggingFace client initialized with API token")
                    else:
                        client = InferenceClient()  # Try without token (rate-limited but should work)
                        if progress_callback:
                            progress_callback("⚠️ HuggingFace client initialized without token (rate-limited)")
                        st.warning("⚠️ No HUGGINGFACE_API_TOKEN found. Add it to .env for better rate limits. Get free token: https://huggingface.co/settings/tokens")
                    
                    if progress_callback:
                        progress_callback(f"🔍 DEBUG: Client type: {type(client).__name__}")
                        
                except Exception as client_err:
                    error_type = type(client_err).__name__
                    error_msg = str(client_err)
                    if progress_callback:
                        progress_callback(f"🔍 DEBUG: Client initialization failed - Type: {error_type}, Error: {error_msg}")
                    st.error(f"❌ Failed to initialize HuggingFace client: {error_type}: {error_msg}")
                    st.info("💡 **Troubleshooting:**\n"
                           "1. Check your internet connection\n"
                           "2. Verify huggingface_hub is installed: `pip install huggingface_hub`\n"
                           "3. Try switching to 'Ollama Local' model\n"
                           "4. Check your HUGGINGFACE_API_TOKEN in .env file")
                    return None
                
                last_error = None
                debug_print(f"Will try {len(models_to_try)} models: {', '.join(models_to_try)}")
                if progress_callback:
                    progress_callback(f"🔍 DEBUG: Will try {len(models_to_try)} models: {', '.join(models_to_try)}")
                
                for idx, model_name in enumerate(models_to_try):
                    try:
                        debug_print(f"[{idx+1}/{len(models_to_try)}] Trying model: {model_name}")
                        if progress_callback:
                            progress_callback(f"🔍 DEBUG: [{idx+1}/{len(models_to_try)}] Trying model: {model_name}")
                        
                        # Format prompt based on model type - use very simple prompts
                        if "flan-t5" in model_name.lower():
                            # T5 models work better with task-based prompts - keep it very simple
                            model_prompt = f"summarize: {transcript_text[:1000]}"
                        elif "gpt2" in model_name.lower() or "distilgpt2" in model_name.lower() or "dialo" in model_name.lower():
                            # GPT-2 and DialoGPT models need very simple prompts - just the text
                            model_prompt = transcript_text[:800]  # Just the text, no instruction
                        else:
                            # Use simple prompt for other models
                            model_prompt = f"Summarize: {transcript_text[:1000]}"
                        
                        # Truncate if needed
                        if len(model_prompt) > MAX_LENGTH:
                            model_prompt = model_prompt[:MAX_LENGTH]
                        
                        # Try text generation (only for models that support text-generation task)
                        try:
                            # For debugging: log the prompt length
                            debug_print(f"Calling {model_name} with prompt length: {len(model_prompt)} chars")
                            debug_print(f"Prompt preview: {model_prompt[:200]}...")
                            if progress_callback:
                                progress_callback(f"🔍 DEBUG: Calling {model_name} with prompt length: {len(model_prompt)} chars")
                                progress_callback(f"🔍 DEBUG: Prompt preview: {model_prompt[:200]}...")
                            
                            debug_print(f"Calling client.text_generation() with model={model_name}, max_new_tokens=256")
                            if progress_callback:
                                progress_callback(f"🔍 DEBUG: Calling client.text_generation() with model={model_name}, max_new_tokens=256")
                            
                            # Call text_generation - this might return a generator or string
                            debug_print(f"About to call text_generation() for {model_name}...")
                            if progress_callback:
                                progress_callback(f"🔍 DEBUG: About to call text_generation()...")
                            
                            # Try with different parameters to avoid empty responses
                            debug_print(f"Calling text_generation with: model={model_name}, prompt_len={len(model_prompt)}, max_tokens=128")
                            
                            # Use smaller max_new_tokens and different temperature for better results
                            # Note: do_sample might not be supported by all models
                            try:
                                response = client.text_generation(
                                    prompt=model_prompt,
                                    model=model_name,
                                    max_new_tokens=128,  # Reduced further to avoid empty responses
                                    temperature=0.5,  # Lower temperature for more focused output
                                    return_full_text=False
                                )
                            except TypeError:
                                # If do_sample or other params not supported, try without them
                                debug_print(f"Retrying without optional parameters for {model_name}")
                                response = client.text_generation(
                                    prompt=model_prompt,
                                    model=model_name,
                                    max_new_tokens=128,
                                    return_full_text=False
                                )
                            
                            debug_print(f"text_generation() returned for {model_name}, type: {type(response).__name__}")
                            if progress_callback:
                                progress_callback(f"🔍 DEBUG: text_generation() returned, type: {type(response).__name__}")
                                
                            # Handle generator responses (some models return generators)
                            # StopIteration can occur when iterating over generators
                            if hasattr(response, '__iter__') and not isinstance(response, (str, bytes)):
                                debug_print(f"Response is iterable (not string), type: {type(response).__name__}")
                                if progress_callback:
                                    progress_callback(f"🔍 DEBUG: Response is iterable (generator), converting to string...")
                                
                                # Safely convert generator/iterator to string
                                try:
                                    # Check if it's a generator (has __next__)
                                    if hasattr(response, '__next__'):
                                        debug_print("Response is a generator, consuming chunks...")
                                        chunks = []
                                        try:
                                            while True:
                                                chunk = next(response)
                                                chunks.append(str(chunk))
                                                debug_print(f"Got chunk: {str(chunk)[:50]}...")
                                        except StopIteration:
                                            # Generator exhausted - this is normal
                                            debug_print(f"Generator exhausted after {len(chunks)} chunks")
                                            response = ''.join(chunks) if chunks else ""
                                            debug_print(f"Final response length: {len(response)}")
                                    else:
                                        # It's a list or other iterable
                                        debug_print("Response is list-like iterable")
                                        response = ''.join(str(item) for item in response)
                                        
                                except StopIteration as si_err:
                                    debug_print(f"StopIteration during processing: {str(si_err)}")
                                    if progress_callback:
                                        progress_callback(f"🔍 DEBUG: StopIteration - generator exhausted")
                                    # Use whatever we collected, or empty string
                                    response = ''.join(chunks) if 'chunks' in locals() and chunks else ""
                                except Exception as gen_err:
                                    debug_print(f"Error processing generator: {type(gen_err).__name__}: {str(gen_err)}")
                                    if progress_callback:
                                        progress_callback(f"🔍 DEBUG: Error processing generator: {type(gen_err).__name__}: {str(gen_err)}")
                                    raise
                            
                            # Debug: log response type and content
                            if progress_callback:
                                response_type = type(response).__name__
                                response_str = str(response) if response else "None"
                                response_preview = response_str[:200] if len(response_str) > 200 else response_str
                                progress_callback(f"🔍 DEBUG: Response received - Type: {response_type}, Length: {len(response_str) if response_str != 'None' else 0}, Preview: {response_preview}")
                                
                        except StopIteration as si_err:
                            # Handle StopIteration specifically - this happens with generators
                            debug_print(f"StopIteration exception for {model_name}: {str(si_err)}")
                            import traceback
                            debug_print(f"Traceback:\n{traceback.format_exc()}")
                            
                            if progress_callback:
                                progress_callback(f"❌ StopIteration error for {model_name} - model returned empty generator")
                            
                            last_error = f"Model {model_name} returned empty response (StopIteration). This usually means:\n- The model couldn't generate text for this prompt\n- The generator was exhausted\n- Try a different model or simpler prompt\n\nFull error: {str(si_err)}"
                            continue
                            
                        except Exception as api_err:
                            # Capture API-specific errors with detailed diagnostics
                            error_str = str(api_err)
                            error_type = type(api_err).__name__
                            import traceback
                            error_traceback = traceback.format_exc()
                            
                            # Log to console
                            debug_print(f"Exception caught for {model_name}")
                            debug_print(f"Exception type: {error_type}")
                            debug_print(f"Exception message: {error_str}")
                            debug_print(f"Full traceback:\n{error_traceback}")
                            
                            # Log full error for debugging (no truncation in progress)
                            if progress_callback:
                                progress_callback(f"🔍 DEBUG: Exception caught for {model_name}")
                                progress_callback(f"🔍 DEBUG: Exception type: {error_type}")
                                progress_callback(f"🔍 DEBUG: Exception message: {error_str}")
                                progress_callback(f"🔍 DEBUG: Full traceback:\n{error_traceback}")
                                progress_callback(f"❌ API Error for {model_name} ({error_type}): {error_str}")
                            
                            # Check for specific error types and provide helpful messages
                            # Always include full error for debugging
                            if "503" in error_str or "Service Unavailable" in error_str:
                                last_error = f"Model {model_name} is loading (503). Wait 30-60 seconds and try again.\n\nFull error ({error_type}): {error_str}"
                            elif "429" in error_str or "rate limit" in error_str.lower():
                                last_error = f"Rate limit exceeded for {model_name}. Add HUGGINGFACE_API_TOKEN to .env or wait a few minutes.\n\nFull error ({error_type}): {error_str}"
                            elif "401" in error_str or "Unauthorized" in error_str:
                                last_error = f"Authentication failed for {model_name}. Check HUGGINGFACE_API_TOKEN in .env file.\n\nFull error ({error_type}): {error_str}"
                            elif "doesn't support task" in error_str.lower() or "not found" in error_str.lower():
                                last_error = f"Model {model_name} doesn't support text-generation or not available.\n\nFull error ({error_type}): {error_str}"
                            elif "timeout" in error_str.lower() or "connection" in error_str.lower():
                                last_error = f"Connection timeout for {model_name}. Check your internet connection.\n\nFull error ({error_type}): {error_str}"
                            else:
                                # Show full error with type for better debugging
                                last_error = f"Error with {model_name} ({error_type}): {error_str}"
                            continue
                        
                        # Clean up response - handle different response types
                        if progress_callback:
                            progress_callback(f"🔍 DEBUG: Processing response for {model_name}...")
                        
                        summary = None
                        if response is not None:
                            if progress_callback:
                                progress_callback(f"🔍 DEBUG: Response is not None, type: {type(response).__name__}")
                            
                            # Response might be a string or have different structure
                            if isinstance(response, str):
                                summary = response.strip()
                                if progress_callback:
                                    progress_callback(f"🔍 DEBUG: Response is string, length: {len(summary)}")
                            elif isinstance(response, list) and len(response) > 0:
                                # Sometimes response is a list
                                if progress_callback:
                                    progress_callback(f"🔍 DEBUG: Response is list with {len(response)} items")
                                summary = str(response[0]).strip()
                            elif hasattr(response, 'generated_text'):
                                if progress_callback:
                                    progress_callback(f"🔍 DEBUG: Response has 'generated_text' attribute")
                                summary = str(response.generated_text).strip()
                            elif hasattr(response, 'text'):
                                if progress_callback:
                                    progress_callback(f"🔍 DEBUG: Response has 'text' attribute")
                                summary = str(response.text).strip()
                            elif hasattr(response, 'content'):
                                if progress_callback:
                                    progress_callback(f"🔍 DEBUG: Response has 'content' attribute")
                                summary = str(response.content).strip()
                            else:
                                # Try to convert to string
                                if progress_callback:
                                    progress_callback(f"🔍 DEBUG: Converting response to string (type: {type(response).__name__})")
                                summary = str(response).strip()
                            
                            # Debug: log what we received
                            if progress_callback:
                                if summary:
                                    progress_callback(f"🔍 DEBUG: Final summary length: {len(summary)} chars")
                                    progress_callback(f"🔍 DEBUG: Summary preview: '{summary[:100]}...'")
                                else:
                                    progress_callback(f"🔍 DEBUG: Summary is empty after processing")
                            
                            # Ensure we got a real response (not just whitespace or error messages)
                            # Reduced minimum length from 10 to 5 to catch shorter valid responses
                            if summary and len(summary.strip()) > 5:
                                # Check for error indicators
                                summary_lower = summary.lower().strip()
                                if not (summary_lower.startswith('error') or 
                                       'not found' in summary_lower or 
                                       'unavailable' in summary_lower or
                                       'loading' in summary_lower or
                                       'rate limit' in summary_lower or
                                       summary_lower == ''):
                                    if progress_callback:
                                        progress_callback("✅ Summary generated successfully!")
                                    return summary
                            
                            # Log what we got for debugging
                            debug_msg = f"Model {model_name} returned invalid/empty response"
                            if summary:
                                debug_msg += f": '{summary[:150]}'"
                            else:
                                debug_msg += " (response was None or empty)"
                            if progress_callback:
                                progress_callback(debug_msg)
                            # Store this as the last error for better reporting
                            if not last_error:
                                last_error = f"Empty or invalid response from {model_name}: {summary[:100] if summary else 'None'}"
                        else:
                            # Response is None
                            if progress_callback:
                                progress_callback(f"Model {model_name} returned None (no response)")
                            if not last_error:
                                last_error = f"Model {model_name} returned None (no response)"
                        
                        # Empty or invalid response, try next model
                        if progress_callback:
                            progress_callback(f"Model {model_name} returned empty response, trying next...")
                        continue
                            
                    except Exception as model_err:
                        # Store error for debugging
                        last_error = str(model_err)
                        error_short = str(model_err)[:150]  # Truncate long errors
                        # Try next model
                        if progress_callback:
                            progress_callback(f"Model {model_name} failed: {error_short}...")
                        continue
                
                # If all HuggingFace models fail, show detailed error and strongly recommend Ollama
                if last_error:
                    # Check if it's just StopIteration (empty generators)
                    if "StopIteration" in last_error or "empty response" in last_error.lower():
                        error_msg = f"⚠️ All HuggingFace text-generation models returned empty responses.\n\n**Why this happens:**\n- HuggingFace text-generation models (gpt2, flan-t5) aren't designed for summarization\n- They return empty generators when they can't generate appropriate text\n- These models work better for completion tasks, not summarization\n\n✅ **Solution:** Switch to 'Ollama Local' - it's specifically designed for tasks like summarization!"
                    elif "503" in last_error or "Service Unavailable" in last_error:
                        error_msg = "HuggingFace service is temporarily unavailable (503). Models may be loading or rate-limited."
                    elif "429" in last_error or "rate limit" in last_error.lower():
                        error_msg = "HuggingFace rate limit exceeded. Please wait a moment and try again."
                    elif "401" in last_error or "Unauthorized" in last_error:
                        error_msg = "HuggingFace authentication failed. Check your HUGGINGFACE_API_TOKEN in .env file."
                    elif "404" in last_error or "not found" in last_error.lower():
                        error_msg = f"Model not found or unavailable. Error: {last_error[:200]}"
                    else:
                        # Show full error (not truncated) for better debugging
                        error_msg = f"All HuggingFace models failed.\n\nFull error details: {last_error}"
                else:
                    error_msg = "All HuggingFace models failed (no valid response received).\n\n**Why:** HuggingFace text-generation models aren't ideal for summarization tasks.\n\n✅ **Solution:** Use 'Ollama Local' model instead!"
                
                st.error(error_msg)
                st.warning("🔄 **Strongly Recommended:** Switch to 'Ollama Local' model - it's much better for summarization!")
                st.info("💡 **Why Ollama is better:**\n"
                       "- ✅ Instruction-tuned models (Mistral, Llama) work excellently for summarization\n"
                       "- ✅ No API rate limits or empty responses\n"
                       "- ✅ Processes locally, more reliable\n"
                       "- ✅ Better multilingual support\n\n"
                       "**To use:** Select 'Ollama Local' in the radio button above")
                
                debug_print("All HuggingFace models failed. User should switch to Ollama Local.")
                return None
        
        # Use Ollama (local) if selected
        if use_local_model:
            if not OLLAMA_AVAILABLE or ChatOllama is None:
                raise ImportError("Ollama not available. Please install: pip install langchain-ollama and ensure Ollama is running.")
            
            if progress_callback:
                progress_callback("Using local Ollama model...")
            
            # Initialize Ollama with available model
            # Try multilingual models first (better for non-English transcripts)
            models_to_try_ollama = [
                "mistral:latest",       # Excellent multilingual capabilities
                "llama3.1:latest",      # Fallback
                "gemma:2b",            # Fast but less multilingual
            ]
            
            llm = None
            model_name = None
            num_ctx = 2048
            
            for ollama_model in models_to_try_ollama:
                try:
                    if progress_callback:
                        progress_callback(f"Trying Ollama model: {ollama_model}...")
                    llm = ChatOllama(
                        model=ollama_model,
                        temperature=0.3,
                        num_ctx=num_ctx
                    )
                    model_name = ollama_model
                    if progress_callback:
                        progress_callback(f"✅ Loaded {ollama_model}")
                    break
                except Exception as model_err:
                    if progress_callback:
                        progress_callback(f"Model {ollama_model} not available, trying next...")
                    continue
            
            if llm is None:
                raise ImportError("No Ollama models available. Please install at least one: llama3.2, mistral, llama3.1, or gemma:2b")
        
            # Chunk transcript if too long (for local models) - better than truncating
            # Note: This should already be handled by the chunking logic above, but keep as fallback
            MAX_TRANSCRIPT_LENGTH = 3000  # Increased threshold
            original_length = len(transcript_text)
            
            # Only chunk if not already chunked and still too long
            if not _is_chunk and original_length > MAX_TRANSCRIPT_LENGTH:
                debug_print(f"Transcript is long ({original_length} chars) for Ollama, chunking...")
                if progress_callback:
                    progress_callback(f"📝 Long transcript ({original_length} chars). Chunking for Ollama...")
                
                # Chunk the transcript
                chunks = chunk_text(transcript_text, chunk_size=2500, overlap=300)
                debug_print(f"Created {len(chunks)} chunks for Ollama")
                if progress_callback:
                    progress_callback(f"📝 Created {len(chunks)} chunks")
                
                # Summarize each chunk
                chunk_summaries = []
                for idx, chunk in enumerate(chunks):
                    debug_print(f"Ollama: Summarizing chunk {idx+1}/{len(chunks)}")
                    if progress_callback:
                        progress_callback(f"📝 Summarizing chunk {idx+1}/{len(chunks)} with Ollama...")
                    
                    # Create prompt for this chunk
                    chunk_prompt = prompt + "\n\nTranscript:\n" + chunk
                    
                    # Generate summary for this chunk
                    chunk_response = llm.invoke(chunk_prompt)
                    
                    # Extract text from response
                    if hasattr(chunk_response, 'content'):
                        chunk_summary = chunk_response.content
                    else:
                        chunk_summary = str(chunk_response)
                    
                    if chunk_summary and len(chunk_summary.strip()) > 10:
                        chunk_summaries.append(chunk_summary.strip())
                        debug_print(f"Chunk {idx+1} summarized: {len(chunk_summary)} chars")
                    else:
                        debug_print(f"Chunk {idx+1} returned empty summary")
                        chunk_summaries.append(f"[Chunk {idx+1} summary unavailable]")
                
                # Combine chunk summaries
                if chunk_summaries:
                    combined_summaries = "\n\n".join([f"**Part {i+1}:**\n{summary}" for i, summary in enumerate(chunk_summaries)])
                    debug_print(f"Combined {len(chunk_summaries)} Ollama chunk summaries")
                    
                    if progress_callback:
                        progress_callback(f"✅ Combined {len(chunk_summaries)} chunk summaries")
                    
                    # If combined summary is still very long, create a final summary
                    if len(combined_summaries) > 5000:
                        if progress_callback:
                            progress_callback("📝 Creating final summary of combined summaries...")
                        final_prompt = prompt + "\n\nTranscript:\n" + combined_summaries
                        final_response = llm.invoke(final_prompt)
                        if hasattr(final_response, 'content'):
                            return final_response.content
                        else:
                            return str(final_response)
                    else:
                        return combined_summaries
                else:
                    debug_print("All Ollama chunk summaries failed")
                    return None
            
            # Combine prompt and transcript
            full_prompt = prompt + "\n\nTranscript:\n" + transcript_text
            
            if progress_callback:
                progress_callback("Generating summary with Ollama (this may take a minute)...")
            
            # Generate summary
            response = llm.invoke(full_prompt)
            
            # Extract text from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
            
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        if use_local_model:
            st.info("Make sure Ollama is running. For faster processing, uncheck 'Use local Ollama model' to use HuggingFace cloud models.")
        else:
            st.info("HuggingFace API might be rate-limited. Try again in a moment, or check 'Use local Ollama model' for local processing.")
        return None

st.title("YouTube Transcript to Notes")
st.caption("Choose your preferred AI model provider")

# Model selection: Let user choose between HuggingFace and Ollama
model_provider = st.radio(
    "Select AI Model Provider:",
    options=["HuggingFace Cloud (Fast)", "Ollama Local"],
    index=0,  # Default to HuggingFace
    help="HuggingFace: Fast cloud models (requires internet, excellent multilingual support). Ollama: Local models (private, slower). For non-English videos, HuggingFace or Ollama with Mistral/Llama3.2 work best."
)

# Determine which provider to use
use_local_model = (model_provider == "Ollama Local")

youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    # Extract video ID for thumbnail display
    try:
        if "youtube.com/watch?v=" in youtube_link:
            video_id = youtube_link.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_link:
            video_id = youtube_link.split("youtu.be/")[1].split("?")[0]
        elif "youtube.com/embed/" in youtube_link:
            video_id = youtube_link.split("embed/")[1].split("?")[0]
        else:
            video_id = youtube_link.split("=")[-1].split("&")[0]
        
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", width=True)
    except Exception as e:
        st.warning("Could not extract video thumbnail")

if st.button("Get Transcript Summary"):
    if not youtube_link:
        st.warning("Please enter a YouTube video link")
    else:
        with st.spinner("Extracting transcript..."):
            transcript_text, used_language = extract_transcript_details(youtube_link)

        if transcript_text:
            # Get language-appropriate prompt
            language_name = used_language.get('name', 'English') if used_language else 'English'
            prompt = get_prompt(language_name)
            
            # Create progress container
            progress_container = st.empty()
            
            def update_progress(message):
                progress_container.info(f"🔄 {message}")
            
            model_display = "Ollama Local" if use_local_model else "HuggingFace Cloud"
            with st.spinner(f"Generating summary with {model_display} ({language_name})..."):
                summary = generate_summary_content(transcript_text, prompt, progress_callback=update_progress, use_local_model=use_local_model)
            
            # Clear progress container
            progress_container.empty()
            
            if summary:
                st.markdown("## Detailed Notes:")
                st.write(summary)
                
                # Show transcript preview
                with st.expander("View Full Transcript"):
                    st.text(transcript_text[:1000] + "..." if len(transcript_text) > 1000 else transcript_text)
            else:
                st.error("Failed to generate summary. Please try again or switch to Ollama Local model.")
        else:
            st.error("Failed to extract transcript. Please check if the video has captions enabled.")


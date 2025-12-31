# YouTube Transcript Summarizer

An intelligent web application that extracts transcripts from YouTube videos and generates comprehensive summaries using AI models. Supports both cloud-based (HuggingFace) and local (Ollama) LLM providers with multilingual support and intelligent chunking for long transcripts.

# Note & Credits:

The code is not productionized version and purely for academic learning purposes. I have used many youtube and github resources to build this code 

## 📋 Table of Contents

- [Overview](#overview)
- [Use Case](#use-case)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture Details](#architecture-details)
- [Design Decisions](#design-decisions)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

This application solves the problem of quickly understanding long YouTube video content without watching the entire video. It automatically:

1. **Extracts transcripts** from YouTube videos (supports multiple languages)
2. **Processes long transcripts** using intelligent chunking (handles 50K+ character transcripts)
3. **Generates summaries** using AI models (HuggingFace cloud or Ollama local)
4. **Preserves context** through overlapping chunks and hierarchical summarization

---

## 💼 Use Case

### Problem Statement
- **Time-consuming**: Watching long YouTube videos (1-3 hours) to extract key information
- **Language barriers**: Videos in non-English languages are hard to understand
- **Information overload**: Long transcripts are difficult to parse manually
- **Accessibility**: Need quick summaries for research, learning, or content creation

### Solution
This application provides:
- ✅ **Fast summarization** of any YouTube video with transcripts
- ✅ **Multilingual support** - summarizes in the video's original language
- ✅ **Handles long videos** - intelligently chunks and processes 50K+ character transcripts
- ✅ **Multiple AI providers** - Choose between cloud (fast) or local (private) models
- ✅ **Comprehensive summaries** - Detailed notes with key points, topics, and insights

### Target Users
- **Students**: Quickly summarize educational videos and lectures
- **Researchers**: Extract key information from long-form content
- **Content Creators**: Understand competitor videos or research topics
- **Language Learners**: Get summaries of videos in foreign languages
- **Professionals**: Stay updated with industry content without watching full videos

---

## ✨ Features

### Core Features
- 🎥 **YouTube Transcript Extraction**
  - Supports multiple YouTube URL formats
  - Automatic language detection
  - Falls back to available languages if preferred language not available
  - Handles videos with/without English transcripts

- 📝 **Intelligent Summarization**
  - Automatic chunking for long transcripts (>3000 chars)
  - Overlapping chunks preserve context
  - Hierarchical summarization for very long content
  - Handles transcripts of any length (tested with 50K+ chars)

- 🌍 **Multilingual Support**
  - Detects transcript language automatically
  - Generates summaries in the same language as transcript
  - Works with Ollama models (Mistral, Llama) for better multilingual support

- 🤖 **Multiple AI Providers**
  - **HuggingFace Cloud**: Fast, cloud-based models (requires internet)
  - **Ollama Local**: Private, local models (works offline)
  - User can choose provider via radio button

- 📊 **Smart Processing**
  - Progress indicators for long operations
  - Debug logging for troubleshooting
  - Error handling with helpful messages
  - Automatic fallback mechanisms

### Advanced Features
- **Chunking Strategy**: Splits long transcripts intelligently with overlap
- **Sentence Boundary Detection**: Breaks chunks at natural sentence endings
- **Recursive Summarization**: Creates final summary of combined chunk summaries
- **Model Selection**: Automatically tries multiple models for best results

---

## 🏗️ Technical Architecture

### High-Level Architecture

```
┌─────────────────┐
│   User Input    │
│ (YouTube URL)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  YouTube Transcript API  │
│  - Extract transcript    │
│  - Detect language       │
│  - Handle errors         │
└────────┬─────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Transcript Processing  │
│  - Check length          │
│  - Chunk if needed       │
│  - Apply overlap         │
└────────┬─────────────────┘
         │
         ▼
┌─────────────────────────┐
│   AI Model Selection     │
│  ┌───────────────────┐  │
│  │ HuggingFace Cloud │  │
│  └───────────────────┘  │
│  ┌───────────────────┐  │
│  │   Ollama Local    │  │
│  └───────────────────┘  │
└────────┬─────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Summarization Engine   │
│  - Process chunks        │
│  - Combine summaries     │
│  - Generate final summary│
└────────┬─────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Summary  │
│  (Display to UI) │
└─────────────────┘
```

### Component Breakdown

#### 1. **Transcript Extraction Module**
- **Library**: `youtube-transcript-api`
- **Functionality**:
  - Extracts video ID from various URL formats
  - Lists available transcript languages
  - Fetches transcript in preferred or available language
  - Handles API version differences (v1.0.0+ vs older)

#### 2. **Chunking Engine**
- **Algorithm**: Fixed-size chunks with overlap
- **Parameters**:
  - Chunk size: 2500 characters
  - Overlap: 300 characters
  - Threshold: 3000 characters (triggers chunking)
- **Features**:
  - Sentence boundary detection
  - Context preservation through overlap
  - Prevents recursive chunking

#### 3. **AI Model Providers**

**HuggingFace Cloud:**
- Models tried (in order):
  1. `google/flan-t5-base` - Reliable, small model
  2. `google/flan-t5-large` - Better quality
  3. `microsoft/DialoGPT-small` - Alternative
  4. `distilgpt2` - Fast fallback
- **API**: HuggingFace Inference API
- **Authentication**: Optional (better rate limits with token)

**Ollama Local:**
- Models tried (in order):
  1. `mistral:latest` - Excellent multilingual support
  2. `llama3.1:latest` - Good quality
  3. `gemma:2b` - Fast fallback
- **Library**: `langchain-ollama`
- **Requirements**: Ollama installed and running locally

#### 4. **Summarization Pipeline**

```
Long Transcript (>3000 chars)
    │
    ├─→ Chunk into pieces (2500 chars, 300 overlap)
    │
    ├─→ Summarize each chunk
    │   ├─→ Chunk 1 → Summary 1
    │   ├─→ Chunk 2 → Summary 2
    │   └─→ Chunk N → Summary N
    │
    ├─→ Combine chunk summaries
    │
    └─→ If combined > 5000 chars:
        └─→ Create final summary of summaries
```

#### 5. **User Interface**
- **Framework**: Streamlit
- **Components**:
  - Model provider selection (radio button)
  - YouTube URL input
  - Video thumbnail display
  - Progress indicators
  - Summary display with expandable transcript view

---

## 📦 Prerequisites

### Required
- **Python 3.8+**
- **pip** (Python package manager)
- **Internet connection** (for HuggingFace cloud models and YouTube API)

### Optional (for local models)
- **Ollama** installed and running
  - Download: https://ollama.com
  - Install models: `ollama pull mistral:latest` or `ollama pull llama3.1:latest`

### API Keys (Optional but Recommended)
- **HuggingFace API Token** (for better rate limits)
  - Get free token: https://huggingface.co/settings/tokens
  - Add to `.env` file: `HUGGINGFACE_API_TOKEN=your_token_here`

---

## 🚀 Installation & Setup

### Step 1: Clone or Navigate to Project

```bash
cd "End To End Youtube Video Transcribe Summarizer LLM App With Google Gemini Pro"
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Optional: For better HuggingFace rate limits
HUGGINGFACE_API_TOKEN=your_huggingface_token_here
```

**To get HuggingFace token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is enough)
3. Copy and paste into `.env` file

### Step 5: Install Ollama (Optional - for local models)

```bash
# Download and install from https://ollama.com
# Then pull a model:
ollama pull mistral:latest
# or
ollama pull llama3.1:latest
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## ⚙️ Configuration

### Model Selection

**In the UI:**
- Select "HuggingFace Cloud (Fast)" for cloud-based models (requires internet)
- Select "Ollama Local" for local models (works offline, more private)

### Chunking Parameters

You can modify these in `app.py`:

```python
CHUNK_THRESHOLD = 3000  # Chunk if transcript > this
CHUNK_SIZE = 2500       # Size of each chunk
CHUNK_OVERLAP = 300     # Overlap between chunks
```

### Model Lists

**HuggingFace models** (in `app.py`):
```python
models_to_try = [
    "google/flan-t5-base",
    "google/flan-t5-large",
    "microsoft/DialoGPT-small",
    "distilgpt2",
]
```

**Ollama models** (in `app.py`):
```python
models_to_try_ollama = [
    "mistral:latest",
    "llama3.1:latest",
    "gemma:2b",
]
```

---

## 📖 Usage

### Basic Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Select AI Provider**:
   - Choose "HuggingFace Cloud (Fast)" or "Ollama Local"

3. **Enter YouTube URL**:
   - Paste any YouTube video URL
   - Supports formats:
     - `https://www.youtube.com/watch?v=VIDEO_ID`
     - `https://youtu.be/VIDEO_ID`
     - `https://www.youtube.com/embed/VIDEO_ID`

4. **Click "Get Detailed Notes"**:
   - App extracts transcript
   - Shows progress messages
   - Generates summary

5. **View Results**:
   - Summary displayed in main area
   - Click "View Full Transcript" to see original transcript

### For Long Videos (50K+ characters)

The app automatically:
1. Detects long transcripts
2. Chunks into manageable pieces
3. Summarizes each chunk
4. Combines summaries
5. Creates final summary if needed

**Progress indicators** show:
- Number of chunks created
- Which chunk is being processed
- When summaries are combined

### Multilingual Videos

The app automatically:
1. Detects transcript language
2. Uses that language for summarization
3. Shows language in progress messages

**Note**: Ollama models (Mistral, Llama) have better multilingual support than HuggingFace text-generation models.

---

## 🏛️ Architecture Details

### Data Flow

```
1. User Input (YouTube URL)
   │
   ├─→ Extract Video ID
   │
   ├─→ Fetch Transcript (YouTube Transcript API)
   │   ├─→ List available languages
   │   ├─→ Select preferred/available language
   │   └─→ Return transcript + language info
   │
   ├─→ Check Transcript Length
   │   ├─→ If > 3000 chars: Chunk
   │   └─→ If ≤ 3000 chars: Process directly
   │
   ├─→ Chunking (if needed)
   │   ├─→ Split into 2500-char chunks
   │   ├─→ 300-char overlap between chunks
   │   └─→ Break at sentence boundaries
   │
   ├─→ Summarization
   │   ├─→ For each chunk:
   │   │   ├─→ Format prompt
   │   │   ├─→ Call AI model
   │   │   └─→ Extract summary
   │   │
   │   └─→ Combine chunk summaries
   │
   ├─→ Final Summary (if combined > 5000 chars)
   │   └─→ Summarize combined summaries
   │
   └─→ Display to User
```

### Processing Strategy

#### For Short Transcripts (≤3000 chars)
```
Transcript → AI Model → Summary
```

#### For Long Transcripts (>3000 chars)
```
Transcript
  │
  ├─→ Chunk 1 → Summary 1
  ├─→ Chunk 2 → Summary 2
  ├─→ Chunk 3 → Summary 3
  └─→ ...
      │
      └─→ Combine Summaries
          │
          └─→ If > 5000 chars:
              └─→ Final Summary
```

### Error Handling

1. **Transcript Extraction Errors**:
   - Handles missing transcripts
   - Falls back to available languages
   - Shows helpful error messages

2. **Model Errors**:
   - Tries multiple models automatically
   - Shows specific error messages
   - Suggests switching providers

3. **Chunking Errors**:
   - Validates chunk sizes
   - Handles edge cases
   - Prevents infinite recursion

---

## 🎨 Design Decisions

### Why Chunking Instead of Truncation?

**Problem**: Long transcripts (50K+ chars) exceed model context limits.

**Solution**: Chunking preserves information:
- ✅ Processes entire transcript
- ✅ Preserves context with overlap
- ✅ Combines summaries intelligently
- ✅ No information loss

**Alternative Considered**: Simple truncation
- ❌ Loses information
- ❌ Only processes beginning/end
- ❌ Misses middle content

### Why Multiple Models?

**Problem**: Different models have different capabilities and availability.

**Solution**: Try multiple models:
- ✅ Fallback if one fails
- ✅ Better success rate
- ✅ Works with different API states

### Why Overlap in Chunks?

**Problem**: Context lost at chunk boundaries.

**Solution**: 300-char overlap:
- ✅ Preserves context between chunks
- ✅ Prevents information loss at boundaries
- ✅ Better summary coherence

### Why Sentence Boundary Detection?

**Problem**: Breaking mid-sentence loses meaning.

**Solution**: Break at sentence endings:
- ✅ More natural chunk boundaries
- ✅ Better context preservation
- ✅ Improved summary quality

### Why Two-Level Summarization?

**Problem**: Combined chunk summaries can be very long.

**Solution**: Final summary if > 5000 chars:
- ✅ More concise final output
- ✅ Better for user consumption
- ✅ Maintains key information

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "All HuggingFace models failed"

**Causes:**
- Rate limiting (free tier limits)
- Models loading (first-time use)
- Network issues

**Solutions:**
- ✅ Add `HUGGINGFACE_API_TOKEN` to `.env`
- ✅ Wait 30-60 seconds and try again
- ✅ Switch to "Ollama Local" model
- ✅ Check internet connection

**See**: `HUGGINGFACE_TROUBLESHOOTING.md` for detailed guide

#### 2. "Error extracting transcript"

**Causes:**
- Video has no transcripts
- Transcripts disabled
- Language not available

**Solutions:**
- ✅ Check if video has captions enabled
- ✅ Try a different video
- ✅ App will try available languages automatically

#### 3. "Ollama model not found"

**Causes:**
- Ollama not installed
- Model not pulled
- Ollama service not running

**Solutions:**
```bash
# Install Ollama from https://ollama.com
# Pull a model:
ollama pull mistral:latest
# Ensure Ollama is running:
ollama list
```

#### 4. "Summary is taking too long"

**Causes:**
- Very long transcript
- Many chunks to process
- Local model is slow

**Solutions:**
- ✅ Use HuggingFace Cloud (faster)
- ✅ Wait - long videos take time
- ✅ Check progress messages

#### 5. "Empty or invalid response"

**Causes:**
- Model returned empty generator
- Rate limits
- Model unavailable

**Solutions:**
- ✅ Switch to Ollama Local
- ✅ Add HuggingFace API token
- ✅ Try again in a few minutes

### Debug Mode

The app includes debug logging. Check the terminal/console where Streamlit is running for `🔍 DEBUG:` messages showing:
- Which model is being tried
- Chunk processing status
- API call details
- Error details

---

## 📚 Project Structure

```
End To End Youtube Video Transcribe Summarizer LLM App With Google Gemini Pro/
│
├── app.py                          # Main application file
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── DESIGN_OPTIONS.md              # Alternative design approaches
├── HUGGINGFACE_TROUBLESHOOTING.md # HuggingFace troubleshooting guide
├── OLLAMA_DOWNLOAD_TROUBLESHOOTING.md # Ollama setup guide
├── .env                           # Environment variables (create this)
└── .venv/                         # Virtual environment (created during setup)
```

---

## 🔮 Future Enhancements

### Planned Features
- [ ] **Parallel chunk processing** - Process chunks simultaneously for speed
- [ ] **Topic-based chunking** - Chunk by topics instead of fixed size
- [ ] **Summary export** - Download summaries as PDF/Markdown
- [ ] **Batch processing** - Process multiple videos at once
- [ ] **Custom prompts** - Let users customize summarization style
- [ ] **Summary history** - Save and retrieve previous summaries
- [ ] **RAG integration** - Vector database for Q&A on transcripts
- [ ] **Video metadata** - Extract and display video info (title, duration, etc.)

### Technical Improvements
- [ ] **Async processing** - Use async/await for parallel operations
- [ ] **Caching** - Cache transcripts and summaries
- [ ] **Better error recovery** - Retry mechanisms
- [ ] **Model fine-tuning** - Fine-tune models for summarization
- [ ] **Streaming support** - Process transcripts as they arrive

---

## 🤝 Contributing

This is a learning project. Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share use cases

---

## 📄 License

This project is for educational purposes. Please respect YouTube's Terms of Service and API usage policies.

---

## 🙏 Acknowledgments

- **Streamlit** - For the web framework
- **youtube-transcript-api** - For transcript extraction
- **HuggingFace** - For cloud AI models
- **Ollama** - For local AI models
- **LangChain** - For LLM integration

---

## 📞 Support

For issues or questions:
1. Check `HUGGINGFACE_TROUBLESHOOTING.md` for HuggingFace issues
2. Check `OLLAMA_DOWNLOAD_TROUBLESHOOTING.md` for Ollama setup
3. Check `DESIGN_OPTIONS.md` for architecture alternatives
4. Review debug messages in console for detailed error info

---

## 🎓 Learning Resources

- **Streamlit**: https://docs.streamlit.io
- **LangChain**: https://python.langchain.com
- **HuggingFace**: https://huggingface.co/docs
- **Ollama**: https://ollama.com/docs
- **YouTube Transcript API**: https://github.com/jdepoix/youtube-transcript-api

---

**Built with ❤️ for efficient content consumption**


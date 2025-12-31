# HuggingFace Cloud Troubleshooting Guide

## Common Issues and Solutions

### 1. **"All HuggingFace models failed" - No Valid Response**

**Possible Causes:**
- Models are still loading (first-time use takes 30-60 seconds)
- Rate limits exceeded (free tier has limits)
- Network connectivity issues
- Models returning empty responses

**Solutions:**
- ✅ **Wait 30-60 seconds** and try again (models need to load on first use)
- ✅ **Add HUGGINGFACE_API_TOKEN** to `.env` file for better rate limits
  - Get free token: https://huggingface.co/settings/tokens
  - Add to `.env`: `HUGGINGFACE_API_TOKEN=your_token_here`
- ✅ **Check internet connection**
- ✅ **Try switching to Ollama Local** for more reliable processing

---

### 2. **"503 Service Unavailable" Error**

**Cause:** Model is loading or temporarily unavailable

**Solution:**
- Wait 30-60 seconds and try again
- Models need to "wake up" on first use
- Try a different model from the list

---

### 3. **"429 Rate Limit Exceeded" Error**

**Cause:** Too many requests without API token

**Solutions:**
- ✅ **Get free HuggingFace API token:**
  1. Go to https://huggingface.co/settings/tokens
  2. Create a new token (read access is enough)
  3. Add to your `.env` file: `HUGGINGFACE_API_TOKEN=your_token_here`
  4. Restart the app

- ✅ **Wait a few minutes** before trying again (rate limits reset)

---

### 4. **"401 Unauthorized" Error**

**Cause:** Invalid or missing API token

**Solutions:**
- Check that `HUGGINGFACE_API_TOKEN` is correctly set in `.env`
- Verify the token is valid at https://huggingface.co/settings/tokens
- Make sure there are no extra spaces in the token

---

### 5. **"Model doesn't support task 'text-generation'"**

**Cause:** Selected model doesn't support text generation

**Solution:**
- The code automatically tries multiple models
- If all fail, try switching to "Ollama Local"

---

### 6. **"Connection timeout" or Network Errors**

**Cause:** Internet connectivity issues

**Solutions:**
- Check your internet connection
- Try again in a few moments
- Use "Ollama Local" for offline processing

---

## Quick Diagnostic Steps

### Step 1: Check Installation
```bash
pip install huggingface_hub
```

### Step 2: Check API Token
1. Create `.env` file in project root (if not exists)
2. Add: `HUGGINGFACE_API_TOKEN=your_token_here`
3. Get token from: https://huggingface.co/settings/tokens

### Step 3: Test Connection
Use the "Test HuggingFace Connection" button in the sidebar to diagnose issues.

### Step 4: Check Error Messages
Look at the progress messages - they will show:
- Which model is being tried
- What error occurred
- Specific troubleshooting steps

---

## Recommended Setup

### For Best Results:
1. ✅ **Get free HuggingFace API token** (increases rate limits)
2. ✅ **Use HuggingFace Cloud** for faster processing
3. ✅ **Have Ollama Local as backup** for offline use

### For Offline Use:
- Use "Ollama Local" option
- Install models: `ollama pull mistral:latest` or `ollama pull llama3.2:latest`

---

## Still Not Working?

1. **Check the error message** in the progress/error section
2. **Try the diagnostic button** in the sidebar
3. **Switch to Ollama Local** as a reliable alternative
4. **Check your `.env` file** has the correct token format

---

## Alternative: Use Ollama Local

If HuggingFace continues to have issues, use "Ollama Local":
- ✅ More reliable (no API dependencies)
- ✅ Works offline
- ✅ No rate limits
- ⚠️ Slower processing
- ⚠️ Requires local model installation


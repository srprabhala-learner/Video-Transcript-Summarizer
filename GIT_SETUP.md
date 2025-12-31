# Git Setup and Push Instructions

## Step-by-Step Commands to Push to GitHub

### Prerequisites
- Git installed on your system
- GitHub account access
- Repository URL: `https://github.com/srprabhala-learner/Video-Transcript-Summarizer`

---

## Commands to Execute

### Step 1: Navigate to Project Directory

```bash
cd "/home/srprabhala/Documents/Learning/LLMProjects/End To End Youtube Video Transcribe Summarizer LLM App With Google Gemini Pro"
```

### Step 2: Initialize Git Repository (if not already initialized)

```bash
git init
```

### Step 3: Configure Git User (if not already configured globally)

```bash
git config user.name "srprabhala-learner"
git config user.email "your-email@example.com"
```

### Step 4: Add Remote Repository

```bash
git remote add origin https://github.com/srprabhala-learner/Video-Transcript-Summarizer.git
```

**If remote already exists, update it:**
```bash
git remote set-url origin https://github.com/srprabhala-learner/Video-Transcript-Summarizer.git
```

### Step 5: Check Remote (Verify)

```bash
git remote -v
```

**Expected output:**
```
origin  https://github.com/srprabhala-learner/Video-Transcript-Summarizer.git (fetch)
origin  https://github.com/srprabhala-learner/Video-Transcript-Summarizer.git (push)
```

### Step 6: Add All Files

```bash
git add .
```

**Or add specific files:**
```bash
git add app.py
git add requirements.txt
git add README.md
git add DESIGN_OPTIONS.md
git add HUGGINGFACE_TROUBLESHOOTING.md
git add OLLAMA_DOWNLOAD_TROUBLESHOOTING.md
git add .gitignore
```

### Step 7: Check Status (Optional - to see what will be committed)

```bash
git status
```

### Step 8: Create Initial Commit

```bash
git commit -m "Initial commit: YouTube Transcript Summarizer with HuggingFace and Ollama support"
```

### Step 9: Set Default Branch (if needed)

```bash
git branch -M main
```

### Step 10: Push to GitHub

**First time push:**
```bash
git push -u origin main
```

**Subsequent pushes:**
```bash
git push
```

---

## Complete Command Sequence (Copy-Paste Ready)

```bash
# Navigate to project
cd "/home/srprabhala/Documents/Learning/LLMProjects/End To End Youtube Video Transcribe Summarizer LLM App With Google Gemini Pro"

# Initialize git (if needed)
git init

# Configure user (if not set globally)
git config user.name "srprabhala-learner"
git config user.email "your-email@example.com"

# Add remote
git remote add origin https://github.com/srprabhala-learner/Video-Transcript-Summarizer.git

# Verify remote
git remote -v

# Add all files
git add .

# Check status
git status

# Commit
git commit -m "Initial commit: YouTube Transcript Summarizer with HuggingFace and Ollama support"

# Set branch name
git branch -M main

# Push to GitHub
git push -u origin main
```

---

## Authentication Options

### Option 1: Personal Access Token (Recommended)

1. **Create Token:**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scopes: `repo` (full control)
   - Copy the token

2. **Use Token:**
   ```bash
   # When prompted for password, use the token instead
   git push -u origin main
   # Username: srprabhala-learner
   # Password: <paste-your-token>
   ```

### Option 2: SSH (Alternative)

1. **Set up SSH key** (if not already done):
   ```bash
   ssh-keygen -t ed25519 -C "your-email@example.com"
   ```

2. **Add SSH key to GitHub:**
   - Copy public key: `cat ~/.ssh/id_ed25519.pub`
   - Add to: https://github.com/settings/keys

3. **Change remote to SSH:**
   ```bash
   git remote set-url origin git@github.com:srprabhala-learner/Video-Transcript-Summarizer.git
   ```

---

## Troubleshooting

### If "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/srprabhala-learner/Video-Transcript-Summarizer.git
```

### If "authentication failed"
- Use Personal Access Token instead of password
- Or set up SSH authentication

### If "branch main has no upstream"
```bash
git push -u origin main
```

### If "failed to push some refs"
```bash
# Pull first (if repository has content)
git pull origin main --allow-unrelated-histories

# Then push
git push -u origin main
```

### If repository is empty on GitHub
- Make sure you've committed files: `git commit -m "message"`
- Check you're pushing to correct branch: `git branch -M main`

---

## Verify Upload

After pushing, check:
1. Go to: https://github.com/srprabhala-learner/Video-Transcript-Summarizer
2. Verify all files are present
3. Check README.md renders correctly

---

## Future Updates

For future changes:

```bash
# Add changes
git add .

# Commit
git commit -m "Description of changes"

# Push
git push
```


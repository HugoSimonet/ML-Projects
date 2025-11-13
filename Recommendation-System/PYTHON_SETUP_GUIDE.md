# Python Version Setup Guide

## Why Switch from Python 3.14?

Python 3.14 is very new and many ML libraries (especially those with C extensions like scikit-surprise) haven't been updated yet. **Python 3.11** is the sweet spot - stable, fast, and fully supported.

---

## Method 1: Install Python 3.11 Alongside Python 3.14 (Recommended)

### Step 1: Download Python 3.11
1. Go to https://www.python.org/downloads/
2. Download **Python 3.11.9** (latest 3.11 version)
3. Run the installer

**IMPORTANT**: During installation:
- ✅ Check "Add Python 3.11 to PATH"
- ✅ Choose "Customize installation"
- ✅ Check "Install for all users"
- ✅ Note the installation path (e.g., `C:\Python311`)

### Step 2: Verify Installation
```bash
# Check Python 3.11 is installed
py -3.11 --version

# Should output: Python 3.11.9
```

### Step 3: Create Virtual Environment with Python 3.11
```bash
# Navigate to your project
cd C:\Users\hugot\Documents\GitHub\ML-Projects\Recommendation-System

# Create virtual environment with Python 3.11
py -3.11 -m venv venv311

# Activate it
venv311\Scripts\activate

# Verify you're using Python 3.11
python --version
# Should show: Python 3.11.9

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Run Your Project
```bash
# With virtual environment activated
python scripts/train_models.py --dataset synthetic --models all --save

# Or run quick start
python examples/quick_start.py
```

---

## Method 2: Use Python Launcher (If Already Installed)

If you already have Python 3.11 installed somewhere:

```bash
# List all Python versions
py --list

# Create venv with specific version
py -3.11 -m venv venv311

# Activate
venv311\Scripts\activate

# Install and run
pip install -r requirements.txt
python scripts/train_models.py
```

---

## Method 3: Use Conda/Miniconda (Alternative)

### Step 1: Install Miniconda
Download from: https://docs.conda.io/en/latest/miniconda.html

### Step 2: Create Environment
```bash
# Create new environment with Python 3.11
conda create -n recsys python=3.11

# Activate it
conda activate recsys

# Navigate to project
cd C:\Users\hugot\Documents\GitHub\ML-Projects\Recommendation-System

# Install dependencies
pip install -r requirements.txt

# Run project
python scripts/train_models.py
```

---

## Method 4: Quick Test Without Installing (Use Python.org Embeddable)

This is temporary but good for testing:

```bash
# Download embeddable Python 3.11 from python.org
# Extract to a folder like C:\Python311Portable

# Use it directly
C:\Python311Portable\python.exe -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
```

---

## Recommended Workflow

### For This Project:
```bash
# 1. Install Python 3.11 from python.org
#    (Keep Python 3.14 - don't uninstall it)

# 2. In your project directory
cd C:\Users\hugot\Documents\GitHub\ML-Projects\Recommendation-System

# 3. Create isolated environment
py -3.11 -m venv venv311

# 4. Activate it
venv311\Scripts\activate

# 5. Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 6. Verify installation
python -c "import surprise; print('Surprise installed!')"

# 7. Run tests
python tests/test_basic.py
python tests/test_integration.py

# 8. Train models
python scripts/train_models.py --dataset synthetic --models all --save

# 9. Run quick start
python examples/quick_start.py
```

---

## Troubleshooting

### Issue: "py -3.11" not found
**Solution**: Python 3.11 not installed or not in PATH
```bash
# Find where Python 3.11 is installed
where python

# Use full path
C:\Python311\python.exe -m venv venv311
```

### Issue: scikit-surprise still won't install
**Solution**: Use pre-built wheel
```bash
# Download wheel from
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-surprise

# Install it
pip install scikit_surprise‑1.1.3‑cp311‑cp311‑win_amd64.whl
```

### Issue: Multiple Python versions confusing
**Solution**: Always use virtual environments
```bash
# Deactivate any active environment
deactivate

# Create fresh environment with specific Python
py -3.11 -m venv fresh_env
fresh_env\Scripts\activate

# Verify
python --version
which python
```

---

## VS Code Setup (If Using)

If you use VS Code:

1. Open Command Palette (`Ctrl+Shift+P`)
2. Search: "Python: Select Interpreter"
3. Choose: `venv311\Scripts\python.exe`
4. Terminal will automatically use correct Python

---

## Quick Reference

### Check Available Python Versions
```bash
py --list
py -0  # Same as above
```

### Create Virtual Environment
```bash
# Python 3.11
py -3.11 -m venv venv311

# Python 3.10
py -3.10 -m venv venv310

# Default Python
python -m venv venv
```

### Activate/Deactivate
```bash
# Activate (Windows)
venv311\Scripts\activate

# Activate (PowerShell)
venv311\Scripts\Activate.ps1

# Deactivate
deactivate
```

### Install Project
```bash
# With venv activated
pip install -r requirements.txt
pip install -e .  # Install in editable mode
```

---

## My Recommendation

**Best approach for your setup:**

1. **Download Python 3.11.9** from python.org
2. **Install** with "Add to PATH" checked
3. **Create venv**: `py -3.11 -m venv venv311`
4. **Activate**: `venv311\Scripts\activate`
5. **Install**: `pip install -r requirements.txt`
6. **Run**: `python scripts/train_models.py`

This keeps Python 3.14 for other projects and gives you 3.11 for ML work.

---

## After Setup

Once you have Python 3.11 environment working:

```bash
# Verify everything works
python tests/test_basic.py
python tests/test_integration.py

# Train a simple model
python scripts/train_models.py --dataset synthetic --models svd --save

# Start API
python -m src.api.app

# Open web demo
# Open web/index.html in browser
```

You'll have the full recommendation system working! 🎉

@echo off
echo ============================================
echo Setting up Recommendation System
echo Using Python 3.12
echo ============================================
echo.

echo Step 1: Creating virtual environment with Python 3.12...
py -3.12 -m venv venv312

echo.
echo Step 2: Activating virtual environment...
call venv312\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo Step 4: Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt

echo.
echo Step 5: Installing project in editable mode...
pip install -e .

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To activate this environment in the future, run:
echo    venv312\Scripts\activate
echo.
echo To test the system, run:
echo    python tests/test_basic.py
echo    python tests/test_integration.py
echo.
echo To train models, run:
echo    python scripts/train_models.py --dataset synthetic --models all --save
echo.
echo To run quick start example:
echo    python examples/quick_start.py
echo.
pause

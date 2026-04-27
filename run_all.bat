@echo off
echo Installing requirements...
pip install -r requirements.txt
echo.
echo Downloading and extracting data...
python download_data.py
echo.
echo Training model...
python train_model.py
echo.
echo Initializing Git repository...
git init
git add .
git commit -m "Initial commit for HAR project"
echo.
echo Done! Please check the output above.
pause

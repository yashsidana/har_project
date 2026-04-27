@echo off
echo Installing requirements...
pip install -r requirements.txt
echo.
echo Downloading and extracting data...
python download_data.py
echo.
echo Training baseline model (Random Forest)...
python train_model.py
echo.
echo Training deep learning models (MLP, CNN, RNN, LSTM, CNN-LSTM)...
python train_dl_model.py
echo.
echo Done! Check loss_curves.png and confusion_matrix.png for results.
pause

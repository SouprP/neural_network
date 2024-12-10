@echo "Checking installed packages / installing"
pip install -q -r "requirements.txt" --no-cache-dir
echo "Running script"
python start.py
pause
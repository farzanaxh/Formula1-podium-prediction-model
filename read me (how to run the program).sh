# 1. Set up virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 2. Install requirements
pip install -r requirements.txt

# 3. Run scripts in order
cd scripts
python 01_data_collection.py
python 02_feature_engineering_fixed.py
python 03_train_model.py
python 04_predict.py
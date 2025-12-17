# 🏠 California House Price Prediction

A modern web application for predicting California housing prices using machine learning. Built with Flask and powered by a Random Forest regression model trained on 20,000+ California housing records.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)

## ✨ Features

- **🎯 Single Prediction**: Enter individual property details for instant price predictions
- **📊 Batch Processing**: Upload CSV files to predict multiple properties at once
- **📈 Data Visualization**: Interactive charts and statistics for batch results
- **💾 Export Results**: Download predictions as CSV files
- **🎨 Premium UI**: Modern dark theme with glassmorphism effects
- **📱 Responsive Design**: Works perfectly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nameisahmedh/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (if not already trained)**
   ```bash
   python main.py
   ```
   This will create `model.pkl` and `pipeline.pkl` files.

4. **Run the Flask application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to: `http://127.0.0.1:5000/`

## 📖 Usage

### Single Prediction

1. Navigate to the "Single Prediction" page
2. Enter property details:
   - Location (longitude, latitude)
   - House median age
   - Room counts (total rooms, bedrooms)
   - Demographics (population, households)
   - Median income
   - Ocean proximity
3. Click "Predict Price" to get instant results

### Batch Prediction

1. Navigate to the "Batch Prediction" page
2. Upload a CSV file with the following columns:
   - `longitude`, `latitude`
   - `housing_median_age`
   - `total_rooms`, `total_bedrooms`
   - `population`, `households`
   - `median_income`
   - `ocean_proximity`
3. View results with statistics and charts
4. Download the results as CSV

### Sample CSV Format

```csv
longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,ocean_proximity
-122.23,37.88,41,880,129,322,126,8.3252,NEAR BAY
-122.22,37.86,21,7099,1106,2401,1138,8.3014,NEAR BAY
```

## 🧠 Model Information

- **Algorithm**: Random Forest Regressor
- **Training Samples**: 20,000+ California housing records
- **Features**: 9 input features including location, demographics, and property details
- **Preprocessing**: 
  - Missing value imputation (median strategy)
  - Standard scaling for numerical features
  - One-hot encoding for categorical features
  - Stratified train/test split based on income categories

## 🛠️ Project Structure

```
California House Prediction/
├── app.py                 # Flask application
├── model_utils.py         # Model utilities and prediction functions
├── main.py               # Original CLI training/inference script
├── requirements.txt      # Python dependencies
├── housing.csv           # Training dataset
├── model.pkl            # Trained model (generated)
├── pipeline.pkl         # Preprocessing pipeline (generated)
├── templates/           # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── batch.html
│   └── error.html
└── static/              # Static assets
    ├── css/
    │   └── style.css
    └── js/
        └── main.js
```

## 🎨 Design

The application features a premium dark theme with:
- Deep purple/blue color scheme
- Glassmorphism effects with backdrop blur
- Smooth animations and transitions
- Responsive grid layouts
- Interactive data visualizations using Chart.js

## 📝 API Endpoints

- `GET /` - Landing page
- `GET /predict` - Single prediction form
- `GET /batch` - Batch prediction page
- `POST /api/predict` - Single prediction API
- `POST /api/batch-predict` - Batch prediction API
- `POST /api/train` - Model training API
- `GET /api/model-status` - Check model status
- `GET /download/<session_id>` - Download results

## 🔧 Development

To modify the model or add features:

1. Edit `model_utils.py` for model logic
2. Update `app.py` for new routes/endpoints
3. Modify templates in `templates/` for UI changes
4. Update `static/css/style.css` for styling

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to fork, modify, and use this project for your own purposes!

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

Built with ❤️ using Flask & Machine Learning

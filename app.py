"""
Flask Web Application for California House Price Prediction
"""
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import uuid
from datetime import datetime
from model_utils import load_model, predict_single, predict_batch, train_model, get_feature_info
try:
    from model_info import get_model_info, load_model_metrics
    MODEL_INFO_AVAILABLE = True
except:
    MODEL_INFO_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'california-house-prediction-2024'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store prediction results temporarily (in production, use a database)
prediction_results = {}


@app.route('/')
def index():
    """Landing page"""
    model, pipeline = load_model()
    model_status = "ready" if model is not None else "not_trained"
    
    # Get model info if available
    model_info = None
    if MODEL_INFO_AVAILABLE and model is not None:
        try:
            model_info = get_model_info()
            # Ensure metrics exist
            if model_info and not model_info.get('metrics'):
                model_info = None
        except Exception as e:
            print(f"Warning: Could not load model info: {e}")
            model_info = None
    
    return render_template('index.html', 
                         model_status=model_status,
                         model_info=model_info)


@app.route('/predict')
def predict_page():
    """Single prediction form page"""
    feature_info = get_feature_info()
    return render_template('predict.html', features=feature_info['features'])


@app.route('/batch')
def batch_page():
    """Batch prediction page"""
    return render_template('batch.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single prediction"""
    try:
        data = request.get_json()
        
        # Convert input to appropriate types
        features = {
            'longitude': float(data.get('longitude')),
            'latitude': float(data.get('latitude')),
            'housing_median_age': float(data.get('housing_median_age')),
            'total_rooms': float(data.get('total_rooms')),
            'total_bedrooms': float(data.get('total_bedrooms')),
            'population': float(data.get('population')),
            'households': float(data.get('households')),
            'median_income': float(data.get('median_income')),
            'ocean_proximity': data.get('ocean_proximity')
        }
        
        result = predict_single(features)
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid input values: {str(e)}"
        }), 400
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        }), 500


@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """API endpoint for batch prediction from CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file uploaded"
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No file selected"
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                "status": "error",
                "message": "Only CSV files are allowed"
            }), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Make predictions
        result = predict_batch(df)
        
        if result['status'] == 'success':
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Save results with proper filename
            result_file = os.path.join(app.config['UPLOAD_FOLDER'], f"predictions_{session_id}.csv")
            result['dataframe'].to_csv(result_file, index=False)
            
            # Store metadata
            prediction_results[session_id] = {
                'timestamp': datetime.now().isoformat(),
                'filename': file.filename,
                'count': result['count'],
                'statistics': result['statistics'],
                'file_path': result_file
            }
            
            return jsonify({
                "status": "success",
                "session_id": session_id,
                "count": result['count'],
                "statistics": result['statistics'],
                "preview": result['dataframe'].head(10).to_dict('records')
            })
        else:
            return jsonify(result), 400
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Processing error: {str(e)}"
        }), 500


@app.route('/api/train', methods=['POST'])
def api_train():
    """API endpoint to trigger model training"""
    try:
        result = train_model()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Training error: {str(e)}"
        }), 500


@app.route('/results/<session_id>')
def results_page(session_id):
    """Display batch prediction results"""
    if session_id not in prediction_results:
        return render_template('error.html', 
                             message="Results not found or expired"), 404
    
    metadata = prediction_results[session_id]
    
    # Read results file
    df = pd.read_csv(metadata['file_path'])
    
    return render_template('results.html',
                         session_id=session_id,
                         metadata=metadata,
                         data=df.to_dict('records'),
                         columns=df.columns.tolist())


@app.route('/download/<session_id>')
def download_results(session_id):
    """Download batch prediction results as CSV"""
    if session_id not in prediction_results:
        return jsonify({
            "status": "error",
            "message": "Results not found"
        }), 404
    
    file_path = prediction_results[session_id]['file_path']
    
    if not os.path.exists(file_path):
        return jsonify({
            "status": "error",
            "message": "File not found"
        }), 404
    
    return send_file(
        file_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"california_predictions_{session_id[:8]}.csv"
    )


@app.route('/download-sample')
def download_sample():
    """Download sample CSV file"""
    sample_path = os.path.join('static', 'sample.csv')
    
    if not os.path.exists(sample_path):
        return jsonify({
            "status": "error",
            "message": "Sample file not found"
        }), 404
    
    return send_file(
        sample_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name='california_housing_sample.csv'
    )


@app.route('/api/model-status')
def model_status():
    """Get current model status"""
    model, pipeline = load_model()
    
    if model is None:
        return jsonify({
            "status": "not_trained",
            "message": "Model has not been trained yet"
        })
    
    return jsonify({
        "status": "ready",
        "message": "Model is ready for predictions",
        "model_type": "RandomForestRegressor"
    })


@app.route('/api/model-info')
def api_model_info():
    """Get detailed model information and metrics"""
    if MODEL_INFO_AVAILABLE:
        try:
            info = get_model_info()
            return jsonify(info)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        model, pipeline = load_model()
        if model is None:
            return jsonify({"status": "not_trained"})
        return jsonify({"status": "ready", "model_name": model.__class__.__name__})


@app.errorhandler(404)
def not_found(e):
    """404 error handler"""
    return render_template('error.html', 
                         message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    """500 error handler"""
    return render_template('error.html',
                         message="Internal server error"), 500


if __name__ == '__main__':
    # Check if model exists
    model, pipeline = load_model()
    if model is None:
        print("\n" + "="*60)
        print("⚠️  WARNING: Model not found!")
        print("="*60)
        print("The trained model doesn't exist yet.")
        print("You can either:")
        print("  1. Train a new model by clicking 'Train Model' in the web UI")
        print("  2. Run 'python main.py' to train via command line")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("✓ Model loaded successfully!")
        print("="*60 + "\n")
    
    print("🚀 Starting Flask server...")
    print("📊 California House Price Prediction Web App")
    print(f"🌐 Open your browser to: http://127.0.0.1:5001/")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

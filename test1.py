import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import requests
from datetime import datetime
import pickle
import base64
from io import BytesIO
import threading

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_session import Session
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
Session(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None
scaler = None
num_classes = None
model_lock = threading.Lock()

# API Keys
OPENWEATHER_API_KEY = "2f45dd6957b064684fce99c5102b53b6"
WEATHERSTACK_API_KEY = "7a9fad52735e3b5fa42c9bc7763a1cd4"

# Disease database (same as before)
DISEASE_DATABASE = {
    "Healthy": {
        "description": "The plant shows no signs of disease.",
        "solution": "Maintain current healthy conditions with proper watering and nutrition.",
        "prevention": "Regular monitoring, proper spacing, and crop rotation.",
        "severity": "None",
        "resources": ["https://extension.umn.edu/plant-diseases", "https://www.apsnet.org/"]
    },
    "Bacterial Spot": {
        "description": "Small, dark, water-soaked spots on leaves that may have yellow halos.",
        "solution": "Apply copper-based bactericides and remove infected plant parts.",
        "prevention": "Use disease-free seeds, avoid overhead watering, and practice crop rotation.",
        "severity": "High",
        "resources": ["https://extension.umn.edu/diseases/bacterial-spot-tomato-and-pepper"]
    },
    "Early Blight": {
        "description": "Concentric rings on leaves with yellow halos, typically starting on older leaves.",
        "solution": "Apply fungicides containing chlorothalonil or mancozeb.",
        "prevention": "Remove plant debris, ensure good air circulation, and practice crop rotation.",
        "severity": "Moderate",
        "resources": ["https://extension.umn.edu/diseases/early-blight-tomato"]
    },
    "Late Blight": {
        "description": "Large, irregularly shaped lesions with white fungal growth under humid conditions.",
        "solution": "Apply fungicides containing metalaxyl or mefenoxam immediately.",
        "prevention": "Avoid overhead watering, remove volunteer plants, and use resistant varieties.",
        "severity": "Critical",
        "resources": ["https://extension.umn.edu/diseases/late-blight-tomato-and-potato"]
    },
    "Leaf Mold": {
        "description": "Yellowish spots on upper leaf surfaces with grayish-purple mold underneath.",
        "solution": "Apply fungicides containing chlorothalonil or copper compounds.",
        "prevention": "Maintain proper humidity levels and ensure good air circulation.",
        "severity": "Moderate",
        "resources": ["https://extension.umn.edu/diseases/leaf-mold-tomato"]
    },
    "Mosaic Virus": {
        "description": "Mottled light and dark green patterns on leaves, often with leaf distortion.",
        "solution": "Remove and destroy infected plants to prevent spread.",
        "prevention": "Control aphids and other insect vectors, use virus-free seeds.",
        "severity": "High",
        "resources": ["https://extension.umn.edu/diseases/tomato-mosaic-virus"]
    },
    "Septoria Leaf Spot": {
        "description": "Small, circular spots with dark borders and light centers on leaves.",
        "solution": "Apply fungicides containing chlorothalonil or mancozeb.",
        "prevention": "Remove infected leaves, avoid overhead watering, and rotate crops.",
        "severity": "Moderate",
        "resources": ["https://extension.umn.edu/diseases/septoria-leaf-spot-tomato"]
    },
    "Spider Mites": {
        "description": "Tiny yellow or white speckles on leaves, fine webbing on undersides.",
        "solution": "Apply miticides or insecticidal soaps, increase humidity.",
        "prevention": "Regularly inspect plants, maintain proper humidity, and remove weeds.",
        "severity": "Moderate",
        "resources": ["https://extension.umn.edu/yard-and-garden-insects/spider-mites"]
    },
    "Target Spot": {
        "description": "Brown spots with concentric rings and yellow halos on leaves.",
        "solution": "Apply fungicides containing chlorothalonil or copper compounds.",
        "prevention": "Remove plant debris, ensure proper spacing, and avoid overhead watering.",
        "severity": "Moderate",
        "resources": ["https://extension.umn.edu/diseases/target-spot-tomato"]
    },
    "Yellow Leaf Curl Virus": {
        "description": "Upward curling of leaves with yellowing and stunted growth.",
        "solution": "Remove infected plants and control whitefly populations.",
        "prevention": "Use resistant varieties and reflective mulches to deter whiteflies.",
        "severity": "Critical",
        "resources": ["https://extension.umn.edu/diseases/tomato-yellow-leaf-curl"]
    }
}

# Utility functions (same as before with minor adaptations)
def save_scaler(scaler, filename='scaler.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(scaler, f)

def load_scaler(filename='scaler.pkl'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def get_weather_openweather(city):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        data = response.json()
        if response.status_code == 200:
            temp = data['main']['temp']
            humidity = data['main']['humidity']
            rainfall = data.get('rain', {}).get('1h', 0)
            if rainfall == 0:
                rainfall = data.get('rain', {}).get('3h', 0) / 3
            return temp, humidity, rainfall
        else:
            raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
    except Exception as e:
        raise Exception(f"OpenWeather API error: {str(e)}")

def get_weather_weatherstack(city):
    try:
        url = f"http://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city}"
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'current' in data:
            temp = data['current']['temperature']
            humidity = data['current']['humidity']
            rainfall = data['current'].get('precip', 0)
            return temp, humidity, rainfall
        else:
            raise Exception(f"API Error: {data.get('error', {}).get('info', 'Unknown error')}")
    except Exception as e:
        raise Exception(f"Weatherstack API error: {str(e)}")

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return tf.clip_by_value(image, 0.0, 1.0)

def create_dataset(X_img, X_env, y, batch_size, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices(({'image_input': X_img, 'env_input': X_env}, y))
    if augment:
        dataset = dataset.map(
            lambda x, y: ({'image_input': augment_image(x['image_input']), 'env_input': x['env_input']}, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    dataset = dataset.shuffle(buffer_size=len(X_img)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model(input_shape_img, input_shape_env, num_classes):
    img_input = Input(shape=input_shape_img, name='image_input')
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    env_input = Input(shape=(input_shape_env,), name='env_input')
    y = Dense(128, activation='relu')(env_input)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    y = Dense(64, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.3)(y)
    
    combined = keras.layers.concatenate([x, y])
    z = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(combined)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    z = Dense(64, activation='relu')(z)
    z = BatchNormalization()(z)
    z = Dropout(0.3)(z)
    z = Dense(num_classes, activation='softmax')(z)
    
    model = Model(inputs=[img_input, env_input], outputs=z)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_data():
    global scaler, num_classes
    try:
        df = pd.read_csv("metadata.csv")
        required_columns = ['image_filename', 'temperature', 'humidity', 'rainfall', 'disease_label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"metadata.csv missing required columns: {required_columns}")
        if df.empty:
            raise ValueError("metadata.csv is empty")
        X_env = df[['temperature', 'humidity', 'rainfall']].values
        y = df['disease_label'].astype('category').cat.codes.values
        num_classes = len(np.unique(y))
        scaler = StandardScaler()
        X_env = scaler.fit_transform(X_env)
        save_scaler(scaler)
        
        stats = {
            'total_samples': len(df),
            'class_distribution': df['disease_label'].value_counts().to_dict(),
            'num_classes': num_classes
        }
        
        return X_env, y, df, stats
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load metadata.csv: {str(e)}")
    except Exception as e:
        raise Exception(f"Error in load_data: {str(e)}")

def load_images(df):
    img_size = (128, 128)
    X_img = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    loaded_count = 0
    for index, row in df.iterrows():
        img_path = os.path.join(base_dir, row['image_filename'])
        if not os.path.exists(img_path):
            continue
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            X_img.append(img)
            loaded_count += 1
        except Exception as e:
            continue
    X_img = np.array(X_img) / 255.0
    if len(X_img) == 0:
        raise ValueError("No images could be loaded. Check file paths and formats.")
    return X_img

def plot_to_base64():
    """Convert matplotlib plot to base64 string for HTML display"""
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_weather', methods=['POST'])
def get_weather():
    city = request.json.get('city', '').strip()
    if not city:
        return jsonify({'error': 'Please enter a city name'}), 400
    
    try:
        try:
            temp, humidity, rainfall = get_weather_openweather(city)
            api_used = "OpenWeatherMap"
        except:
            temp, humidity, rainfall = get_weather_weatherstack(city)
            api_used = "Weatherstack"
        
        return jsonify({
            'success': True,
            'temperature': round(temp, 1),
            'humidity': int(humidity),
            'rainfall': round(rainfall, 1),
            'api_used': api_used
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_model', methods=['POST'])
def train_model():
    def training_task():
        global model, scaler, num_classes
        try:
            X_env, y, df, stats = load_data()
            X_img = load_images(df)
            
            min_samples = min(len(X_env), len(X_img))
            X_env = X_env[:min_samples]
            X_img = X_img[:min_samples]
            y = y[:min_samples]
            
            class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
            class_weight_dict = dict(zip(np.unique(y), class_weights))
            
            X_env_train, X_env_temp, X_img_train, X_img_temp, y_train, y_temp = train_test_split(
                X_env, X_img, y, test_size=0.3, random_state=42, stratify=y)
            X_env_val, X_env_test, X_img_test, y_val, y_test = train_test_split(
                X_env_temp, X_img_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
            
            local_model = build_model(X_img_train.shape[1:], X_env_train.shape[1], num_classes)
            
            batch_size = 32
            train_dataset = create_dataset(X_img_train, X_env_train, y_train, batch_size, augment=True)
            val_dataset = create_dataset(X_img_val, X_env_val, y_val, batch_size, augment=False)
            
            epochs = 100
            history = local_model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                class_weight=class_weight_dict,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, min_delta=0.001),
                    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
                ],
                verbose=0
            )
            
            with model_lock:
                model = local_model
            
            # Save model and labels
            local_model.save('crop_disease_model.keras')
            with open('disease_labels.pkl', 'wb') as f:
                pickle.dump(df['disease_label'].unique(), f)
            
            # Generate plots
            plt.figure(figsize=(12, 4))
            
            plt.subplot(131)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(132)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Confusion Matrix
            plt.subplot(133)
            test_dataset = create_dataset(X_img_test, X_env_test, y_test, batch_size=32, augment=False)
            y_pred = []
            y_true = []
            for batch in test_dataset:
                inputs, labels = batch
                pred = model.predict(inputs, verbose=0)
                y_pred.extend(np.argmax(pred, axis=1))
                y_true.extend(labels.numpy())
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
            
            cm = confusion_matrix(y_true, y_pred)
            diseases = df['disease_label'].unique()
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=diseases, yticklabels=diseases)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            
            training_plot = plot_to_base64()
            plt.close()
            
            # Test evaluation
            test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
            
            session['training_results'] = {
                'success': True,
                'test_accuracy': float(test_acc),
                'test_loss': float(test_loss),
                'training_plot': training_plot,
                'stats': stats
            }
            
        except Exception as e:
            session['training_results'] = {
                'success': False,
                'error': str(e)
            }
    
    threading.Thread(target=training_task).start()
    return jsonify({'message': 'Training started'})

@app.route('/training_status')
def training_status():
    results = session.get('training_results')
    if results:
        session.pop('training_results', None)
        return jsonify(results)
    return jsonify({'status': 'running'})

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    
    if model is None:
        return jsonify({'error': 'Model not trained or loaded'}), 400
    
    try:
        # Get environmental data
        temp = float(request.form.get('temperature'))
        hum = float(request.form.get('humidity'))
        rain = float(request.form.get('rainfall'))
        
        if not (-10 <= temp <= 50):
            return jsonify({'error': 'Temperature should be between -10°C and 50°C'}), 400
        if not (0 <= hum <= 100):
            return jsonify({'error': 'Humidity should be between 0% and 100%'}), 400
        if not (0 <= rain <= 100):
            return jsonify({'error': 'Rainfall should be between 0mm and 100mm'}), 400
        
        # Get image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save uploaded file
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess environmental data
        env_data = np.array([[temp, hum, rain]])
        if scaler is None:
            scaler = load_scaler()
            if scaler is None:
                return jsonify({'error': 'Scaler not found'}), 400
        env_data = scaler.transform(env_data)
        
        # Preprocess image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to load image'}), 400
        
        img = cv2.resize(img, (128, 128)) / 255.0
        
        if request.form.get('augment') == 'true':
            img = augment_image(img)
        
        img_data = np.expand_dims(img, axis=0)
        
        # Make prediction
        with model_lock:
            prediction = model.predict([img_data, env_data], verbose=0)
        
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        
        # Load disease labels
        try:
            with open('disease_labels.pkl', 'rb') as f:
                diseases = pickle.load(f)
        except FileNotFoundError:
            diseases = pd.read_csv("metadata.csv")['disease_label'].unique()
        
        predicted_disease = diseases[predicted_class]
        disease_info = DISEASE_DATABASE.get(predicted_disease, {
            "description": "No information available.",
            "solution": "Consult a specialist.",
            "prevention": "General prevention methods.",
            "severity": "Unknown",
            "resources": []
        })
        
        # Convert image to base64 for display
        _, img_buffer = cv2.imencode('.png', cv2.resize(cv2.imread(filepath), (250, 250)))
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')
        
        # Store in session history
        if 'prediction_history' not in session:
            session['prediction_history'] = []
        
        session['prediction_history'].append({
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image': filename,
            'disease': predicted_disease,
            'confidence': f"{confidence:.2%}"
        })
        
        # Keep only last 10 predictions
        session['prediction_history'] = session['prediction_history'][-10:]
        
        return jsonify({
            'success': True,
            'predicted_disease': predicted_disease,
            'confidence': float(confidence),
            'disease_info': disease_info,
            'image_data': img_base64,
            'all_predictions': {
                disease: float(prediction[0][i]) for i, disease in enumerate(diseases)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    global model, scaler
    
    if model is None:
        return jsonify({'error': 'Model not trained or loaded'}), 400
    
    try:
        temp = float(request.form.get('temperature'))
        hum = float(request.form.get('humidity'))
        rain = float(request.form.get('rainfall'))
        
        # Environmental data validation
        if not (-10 <= temp <= 50) or not (0 <= hum <= 100) or not (0 <= rain <= 100):
            return jsonify({'error': 'Invalid environmental data'}), 400
        
        # Preprocess environmental data
        env_data = np.array([[temp, hum, rain]])
        if scaler is None:
            scaler = load_scaler()
            if scaler is None:
                return jsonify({'error': 'Scaler not found'}), 400
        env_data = scaler.transform(env_data)
        
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No images provided'}), 400
        
        results = []
        for file in files:
            if file.filename == '':
                continue
            
            # Save and process each file
            filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess image
            img = cv2.imread(filepath)
            if img is None:
                continue
            
            img = cv2.resize(img, (128, 128)) / 255.0
            if request.form.get('augment') == 'true':
                img = augment_image(img)
            
            img_data = np.expand_dims(img, axis=0)
            env_data_batch = np.repeat(env_data, 1, axis=0)
            
            # Make prediction
            with model_lock:
                prediction = model.predict([img_data, env_data_batch], verbose=0)
            
            predicted_class = np.argmax(prediction)
            confidence = prediction[0][predicted_class]
            
            # Load disease labels
            try:
                with open('disease_labels.pkl', 'rb') as f:
                    diseases = pickle.load(f)
            except FileNotFoundError:
                diseases = pd.read_csv("metadata.csv")['disease_label'].unique()
            
            predicted_disease = diseases[predicted_class]
            
            results.append({
                'file': file.filename,
                'disease': predicted_disease,
                'confidence': float(confidence)
            })
        
        if not results:
            return jsonify({'error': 'No valid images processed'}), 400
        
        # Create summary
        disease_counts = {}
        for result in results:
            disease = result['disease']
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_images': len(results),
                'disease_distribution': disease_counts
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model_route():
    global model, scaler, num_classes
    
    if 'model_file' not in request.files:
        return jsonify({'error': 'No model file provided'}), 400
    
    file = request.files['model_file']
    if file.filename == '':
        return jsonify({'error': 'No model file selected'}), 400
    
    try:
        # Save uploaded model file
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_model.keras')
        file.save(model_path)
        
        # Load model
        model = load_model(model_path)
        scaler = load_scaler()
        
        if scaler is None:
            return jsonify({'error': 'Scaler file not found'}), 400
        
        # Load disease labels
        try:
            with open('disease_labels.pkl', 'rb') as f:
                diseases = pickle.load(f)
            num_classes = len(diseases)
        except FileNotFoundError:
            diseases = pd.read_csv("metadata.csv")['disease_label'].unique()
            num_classes = len(diseases)
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully with {num_classes} classes'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_summary')
def model_summary():
    if model is None:
        return jsonify({'error': 'No model loaded'}), 400
    
    summary = []
    model.summary(print_fn=lambda x: summary.append(x))
    
    return jsonify({
        'summary': summary,
        'parameters': model.count_params()
    })

@app.route('/prediction_history')
def get_prediction_history():
    history = session.get('prediction_history', [])
    return jsonify({'history': history})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
class CropDiseaseApp {
    constructor() {
        this.initializeEventListeners();
        this.loadPredictionHistory();
    }

    initializeEventListeners() {
        // Weather fetching
        document.getElementById('fetchWeather').addEventListener('click', () => this.fetchWeather());
        
        // Model controls
        document.getElementById('trainModel').addEventListener('click', () => this.trainModel());
        document.getElementById('loadModel').addEventListener('click', () => this.loadModel());
        document.getElementById('modelSummary').addEventListener('click', () => this.showModelSummary());
        
        // Prediction controls
        document.getElementById('singlePredict').addEventListener('click', () => this.triggerSingleImage());
        document.getElementById('batchPredict').addEventListener('click', () => this.triggerBatchImages());
        
        document.getElementById('singleImage').addEventListener('change', (e) => this.predictSingle(e));
        document.getElementById('batchImages').addEventListener('change', (e) => this.predictBatch(e));
    }

    showStatus(message, type = 'info') {
        const statusEl = document.getElementById('statusMessage');
        statusEl.textContent = message;
        statusEl.className = `alert alert-${type}`;
    }

    async fetchWeather() {
        const city = document.getElementById('cityInput').value.trim();
        if (!city) {
            this.showStatus('Please enter a city name', 'warning');
            return;
        }

        this.showStatus('Fetching weather data...', 'info');

        try {
            const response = await fetch('/get_weather', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ city: city })
            });

            const data = await response.json();

            if (data.success) {
                document.getElementById('temperature').value = data.temperature;
                document.getElementById('humidity').value = data.humidity;
                document.getElementById('rainfall').value = data.rainfall;
                this.showStatus(`Weather data fetched from ${data.api_used}`, 'success');
            } else {
                this.showStatus(data.error, 'danger');
            }
        } catch (error) {
            this.showStatus('Failed to fetch weather data', 'danger');
        }
    }

    async trainModel() {
        this.showStatus('Starting model training...', 'info');
        const progressBar = document.getElementById('trainingProgress');
        progressBar.style.display = 'block';

        try {
            const response = await fetch('/train_model', {
                method: 'POST'
            });

            if (response.ok) {
                this.checkTrainingStatus();
            } else {
                this.showStatus('Failed to start training', 'danger');
                progressBar.style.display = 'none';
            }
        } catch (error) {
            this.showStatus('Training failed: ' + error.message, 'danger');
            progressBar.style.display = 'none';
        }
    }

    async checkTrainingStatus() {
        try {
            const response = await fetch('/training_status');
            const data = await response.json();

            if (data.status === 'running') {
                setTimeout(() => this.checkTrainingStatus(), 2000);
                return;
            }

            const progressBar = document.getElementById('trainingProgress');
            progressBar.style.display = 'none';

            if (data.success) {
                this.showStatus('Training completed successfully!', 'success');
                this.displayTrainingResults(data);
            } else {
                this.showStatus('Training failed: ' + data.error, 'danger');
            }
        } catch (error) {
            this.showStatus('Error checking training status', 'danger');
        }
    }

    displayTrainingResults(data) {
        const resultsDiv = document.getElementById('trainingResults');
        const plotImg = document.getElementById('trainingPlot');
        
        resultsDiv.innerHTML = `
            <div class="alert alert-success">
                <h5>Training Completed!</h5>
                <p>Test Accuracy: ${(data.test_accuracy * 100).toFixed(2)}%</p>
                <p>Test Loss: ${data.test_loss.toFixed(4)}</p>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <h6>Dataset Statistics:</h6>
                    <ul>
                        <li>Total Samples: ${data.stats.total_samples}</li>
                        <li>Number of Classes: ${data.stats.num_classes}</li>
                    </ul>
                </div>
            </div>
        `;
        
        plotImg.src = 'data:image/png;base64,' + data.training_plot;
        plotImg.style.display = 'block';
    }

    async loadModel() {
        const fileInput = document.getElementById('modelFile');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showStatus('Please select a model file', 'warning');
            return;
        }

        const formData = new FormData();
        formData.append('model_file', file);

        try {
            const response = await fetch('/load_model', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.showStatus(data.message, 'success');
            } else {
                this.showStatus(data.error, 'danger');
            }
        } catch (error) {
            this.showStatus('Failed to load model', 'danger');
        }
    }

    async showModelSummary() {
        try {
            const response = await fetch('/model_summary');
            const data = await response.json();

            if (data.error) {
                this.showStatus(data.error, 'warning');
                return;
            }

            document.getElementById('modelSummaryText').textContent = 
                data.summary.join('\n') + `\n\nTotal Parameters: ${data.parameters}`;
            
            new bootstrap.Modal(document.getElementById('modelSummaryModal')).show();
        } catch (error) {
            this.showStatus('Failed to get model summary', 'danger');
        }
    }

    triggerSingleImage() {
        document.getElementById('singleImage').click();
    }

    triggerBatchImages() {
        document.getElementById('batchImages').click();
    }

    async predictSingle(event) {
        const file = event.target.files[0];
        if (!file) return;

        const temp = document.getElementById('temperature').value;
        const hum = document.getElementById('humidity').value;
        const rain = document.getElementById('rainfall').value;
        const augment = document.getElementById('augmentation').checked;

        if (!temp || !hum || !rain) {
            this.showStatus('Please fill all environmental data fields', 'warning');
            return;
        }

        this.showStatus('Processing prediction...', 'info');

        const formData = new FormData();
        formData.append('image', file);
        formData.append('temperature', temp);
        formData.append('humidity', hum);
        formData.append('rainfall', rain);
        formData.append('augment', augment);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.showStatus('Prediction completed!', 'success');
                this.displayPredictionResults(data);
                this.loadPredictionHistory();
            } else {
                this.showStatus('Prediction failed: ' + data.error, 'danger');
            }
        } catch (error) {
            this.showStatus('Prediction failed', 'danger');
        }
    }

    async predictBatch(event) {
        const files = event.target.files;
        if (!files.length) return;

        const temp = document.getElementById('temperature').value;
        const hum = document.getElementById('humidity').value;
        const rain = document.getElementById('rainfall').value;
        const augment = document.getElementById('augmentation').checked;

        if (!temp || !hum || !rain) {
            this.showStatus('Please fill all environmental data fields', 'warning');
            return;
        }

        this.showStatus(`Processing ${files.length} images...`, 'info');

        const formData = new FormData();
        for (let file of files) {
            formData.append('images', file);
        }
        formData.append('temperature', temp);
        formData.append('humidity', hum);
        formData.append('rainfall', rain);
        formData.append('augment', augment);

        try {
            const response = await fetch('/batch_predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.showStatus(`Batch prediction completed! Processed ${data.summary.total_images} images`, 'success');
                this.displayBatchResults(data);
                this.loadPredictionHistory();
            } else {
                this.showStatus('Batch prediction failed: ' + data.error, 'danger');
            }
        } catch (error) {
            this.showStatus('Batch prediction failed', 'danger');
        }
    }

    displayPredictionResults(data) {
        // Display image
        const imageEl = document.getElementById('selectedImage');
        const noImageEl = document.getElementById('noImage');
        imageEl.src = 'data:image/png;base64,' + data.image_data;
        imageEl.style.display = 'block';
        noImageEl.style.display = 'none';

        // Display result
        document.getElementById('resultDisplay').innerHTML = `
            <div class="alert alert-success">
                <h5>Prediction Result</h5>
                <p><strong>Disease:</strong> ${data.predicted_disease}</p>
                <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
            </div>
        `;

        // Display disease information
        const diseaseInfo = data.disease_info;
        document.getElementById('diseaseInfo').innerHTML = `
            <h4>${data.predicted_disease}</h4>
            <p><strong>Description:</strong> ${diseaseInfo.description}</p>
            <p><strong>Solution:</strong> ${diseaseInfo.solution}</p>
            <p><strong>Prevention:</strong> ${diseaseInfo.prevention}</p>
            <p><strong>Severity:</strong> ${diseaseInfo.severity}</p>
            ${diseaseInfo.resources.length > 0 ? `
                <p><strong>Resources:</strong></p>
                <ul>
                    ${diseaseInfo.resources.map(resource => 
                        `<li><a href="${resource}" target="_blank">${resource}</a></li>`
                    ).join('')}
                </ul>
            ` : ''}
        `;
    }

    displayBatchResults(data) {
        const resultsDiv = document.getElementById('batchResults');
        
        let html = `
            <div class="alert alert-success">
                <h5>Batch Prediction Summary</h5>
                <p>Total Images Processed: ${data.summary.total_images}</p>
            </div>
            
            <h6>Disease Distribution:</h6>
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Disease</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        Object.entries(data.summary.disease_distribution).forEach(([disease, count]) => {
            const percentage = ((count / data.summary.total_images) * 100).toFixed(1);
            html += `
                <tr>
                    <td>${disease}</td>
                    <td>${count}</td>
                    <td>${percentage}%</td>
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
            
            <h6>Detailed Results:</h6>
            <div class="table-responsive">
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th>File</th>
                            <th>Predicted Disease</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        data.results.forEach(result => {
            html += `
                <tr>
                    <td>${result.file}</td>
                    <td>${result.disease}</td>
                    <td>${(result.confidence * 100).toFixed(2)}%</td>
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;

        resultsDiv.innerHTML = html;
    }

    async loadPredictionHistory() {
        try {
            const response = await fetch('/prediction_history');
            const data = await response.json();

            const historyDiv = document.getElementById('predictionHistory');
            
            if (data.history.length === 0) {
                historyDiv.innerHTML = '<p class="text-center text-muted">No predictions yet</p>';
                return;
            }

            let html = '<div class="table-responsive"><table class="table table-striped"><thead><tr><th>Time</th><th>Image</th><th>Disease</th><th>Confidence</th></tr></thead><tbody>';
            
            data.history.slice().reverse().forEach(pred => {
                html += `
                    <tr>
                        <td>${pred.time}</td>
                        <td>${pred.image}</td>
                        <td>${pred.disease}</td>
                        <td>${pred.confidence}</td>
                    </tr>
                `;
            });
            
            html += '</tbody></table></div>';
            historyDiv.innerHTML = html;
        } catch (error) {
            console.error('Failed to load prediction history:', error);
        }
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new CropDiseaseApp();
});
# ML Platform - No-Code Machine Learning Platform

A comprehensive no-code machine learning platform that allows users to upload datasets, train various ML models, and deploy them with ease. Built with Flask backend and modern web technologies.

![Overview](https://github.com/Reaishma/LUFT-Artificial-Intelligence-/blob/main/Frontend%20files%2Fchrome_screenshot_Sep%2016%2C%202025%2010_11_45%20AM%20GMT%2B05_30.png)

# Live Demo 

**ML Ocean City**-https://reaishma.github.io/LUFT-Artificial-Intelligence-/

## 🚀 Features

### Core Functionality
- **Easy Data Upload**: Support for CSV, Excel (.xlsx), and JSON formats
- **Automated Data Preprocessing**: Handle missing values, normalization, and feature engineering
- **Multiple ML Algorithms**: 
  - **Regression**: Linear Regression, Decision Tree, Random Forest, SVM, Neural Networks
  - **Classification**: Logistic Regression, Decision Tree, Random Forest, SVM, Neural Networks  
  - **Clustering**: K-Means, Hierarchical Clustering
  - **Dimensionality Reduction**: Principal Component Analysis (PCA)

![Model builder](https://github.com/Reaishma/LUFT-Artificial-Intelligence-/blob/main/Frontend%20files%2Fchrome_screenshot_Sep%2016%2C%202025%2010_13_00%20AM%20GMT%2B05_30.png)

- **Model Training**: Background training with real-time progress tracking
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **Model Deployment**: One-click deployment to cloud platforms (AWS, GCP, Azure)

### User Interface
- **Landing Page**: Platform introduction and benefits
- **Model Builder**: Step-by-step model creation wizard
- **Model Gallery**: Browse and manage trained models
- **Data Uploader**: Drag-and-drop data upload with analysis
- **Model Trainer**: Monitor training progress and logs
- **Model Evaluator**: Comprehensive performance analysis
- **Model Deployer**: Deploy models as REST APIs
- **Documentation**: Complete user guides and API documentation
- **Support**: Help center and troubleshooting

## 📁 Project Structure

```
ML-Platform/
├── Frontend Files
│   ├── index.html              # Landing page
│   ├── model-builder.html      # Model builder interface
│   ├── model-gallery.html      # Model gallery
│   ├── data-uploader.html      # Data upload interface
│   ├── model-trainer.html      # Training interface
│   ├── model-evaluator.html    # Evaluation interface
│   ├── model-deployer.html     # Deployment interface
│   ├── documentation.html      # Documentation
│   ├── support.html            # Support center
│   ├── style.css               # Styling
│   └── script.js               # Frontend logic
├── Backend Files
│   ├── app.py                  # Main Flask application
│   ├── models.py               # ML model management
│   ├── data.py                 # Data processing
│   ├── train.py                # Model training
│   ├── evaluate.py             # Model evaluation
│   └── deploy.py               # Model deployment
├── Configuration
│   ├── requirements.txt        # Python dependencies
│   └── README.md              # This file

```

## 🛠 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Modern web browser
- 4GB+ RAM recommended
- Internet connection for cloud deployments

### Quick Start

1. **Clone the repository** (or ensure all files are in your project directory)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Flask server**:
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

### Development Setup

For development with auto-reload:
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

## 🎯 Quick Usage Guide

### 1. Upload Data
- Go to **Data Uploader** page
- Drag & drop or click to upload CSV, Excel, or JSON files
- View automatic data analysis and quality assessment
- Configure preprocessing options

### 2. Build Model
- Navigate to **Model Builder**
- Select your algorithm type (Regression, Classification, Clustering, etc.)
- Choose specific algorithm and configure parameters
- Select dataset and target column
- Create your model

### 3. Train Model
- Monitor training progress in **Model Trainer**
- View real-time logs and status updates
- Control training process (stop, pause, restart)
- Get notified when training completes

### 4. Evaluate Model
- Use **Model Evaluator** for comprehensive performance analysis
- View metrics like accuracy, F1-score, R² score
- Generate confusion matrices and ROC curves
- Compare multiple models

### 5. Deploy Model
- Go to **Model Deployer**
- Select trained model and deployment configuration
- Choose cloud provider (AWS, GCP, Azure) or deploy locally
- Get REST API endpoints for your model

## 📊 Supported Algorithms

### Regression
- **Linear Regression**: Simple linear relationships
- **Decision Tree Regressor**: Non-linear patterns
- **Random Forest Regressor**: Ensemble method, robust
- **Support Vector Regressor**: High-dimensional data
- **Neural Network Regressor**: Complex patterns

### Classification
- **Logistic Regression**: Linear classification
- **Decision Tree Classifier**: Interpretable rules
- **Random Forest Classifier**: High accuracy ensemble
- **Support Vector Classifier**: Margin-based classification
- **Neural Network Classifier**: Deep learning patterns

### Clustering
- **K-Means**: Partition-based clustering
- **Hierarchical Clustering**: Tree-based clustering

### Dimensionality Reduction
- **PCA**: Principal Component Analysis

## 🔧 API Reference

### Health Check
```http
GET /api/health
```

### List Algorithms
```http
GET /api/algorithms
```

### Upload Dataset
```http
POST /api/upload
Content-Type: multipart/form-data

file: [your-dataset-file]
```

### Train Model
```http
POST /api/train
Content-Type: application/json

{
  "model_name": "My Model",
  "algorithm": "random_forest_classifier",
  "dataset_id": "uuid-here",
  "target_column": "target",
  "parameters": {
    "n_estimators": 100,
    "max_depth": 10
  }
}
```

### Check Training Status
```http
GET /api/training-jobs/{job_id}
```

### List Models
```http
GET /api/models
```

### List Datasets
```http
GET /api/datasets
```

## 🚀 Deployment Options

### Local Deployment
- Docker containers
- Direct Flask server
- Development environment

### Cloud Deployment
- **AWS**: ECS, Lambda, EC2
- **Google Cloud**: Cloud Run, App Engine
- **Azure**: Container Instances, App Service

### Features
- Auto-scaling capabilities
- Load balancing
- Health monitoring
- SSL/TLS encryption
- Cost estimation

## 📈 Model Performance Tracking

### Metrics Supported
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC AUC
- **Regression**: R² Score, RMSE, MAE, MAPE
- **Clustering**: Silhouette Score, Inertia
- **Feature Importance**: For tree-based models

### Visualizations
- Confusion matrices
- ROC curves
- Precision-recall curves
- Feature importance plots
- Performance comparison charts

## 🔒 Security Features

- Secure file upload with validation
- Input sanitization
- Error handling and logging
- No exposure of sensitive information
- Cache control for development

## 🛡 Data Privacy

- Local processing by default
- No data sent to external services without consent
- Secure cloud deployment options
- User control over data storage

## 📝 Configuration

### Environment Variables
- `FLASK_ENV`: development/production
- `FLASK_DEBUG`: True/False for debug mode
- `MAX_CONTENT_LENGTH`: Maximum upload size
- Cloud provider credentials for deployment

### Preprocessing Options
- Missing value handling strategies
- Feature scaling methods
- Categorical encoding approaches
- Feature selection techniques

## 🐛 Troubleshooting

### Common Issues

**Upload fails:**
- Check file size (max 100MB)
- Verify file format (CSV, Excel, JSON)
- Ensure proper file structure

**Training fails:**
- Verify data quality
- Check target column exists
- Ensure sufficient data samples
- Review algorithm parameters

**Deployment fails:**
- Check cloud credentials
- Verify network connectivity
- Review deployment configuration
- Check resource limits

### Performance Tips
- Clean data before upload
- Remove irrelevant features
- Start with simple algorithms
- Monitor resource usage
- Use appropriate instance sizes for deployment

## 📞 Support

### Documentation
- Complete user guides in the **Documentation** section
- API reference with examples
- Video tutorials (coming soon)

### Community
- GitHub Issues for bug reports
- Feature requests welcome
- Community forum discussions

### Contact
- Email: vra.9618@gmail.com
- Emergency support: Available for critical issues

![Support](https://github.com/Reaishma/LUFT-Artificial-Intelligence-/blob/main/chrome_screenshot_Sep%2016%2C%202025%2010_32_52%20AM%20GMT%2B05_30.png)

## 🔮 Roadmap

### Upcoming Features
- **Deep Learning**: CNN and RNN support with TensorFlow/PyTorch
- **Time Series**: Specialized algorithms for temporal data
- **Auto ML**: Automated model selection and hyperparameter tuning
- **Advanced Visualizations**: Interactive charts and dashboards
- **Collaborative Features**: Team workspaces and model sharing
- **Model Versioning**: Track model iterations and comparisons
- **Data Connectors**: Direct database and API integrations
- **Real-time Inference**: Streaming predictions
- **Model Monitoring**: Production model performance tracking

### Platform Improvements
- Enhanced mobile interface
- Offline capabilities
- Advanced security features
- Multi-language support
- Enterprise SSO integration

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

### Development Guidelines
- Follow PEP 8 for Python code
- Use semantic commit messages
- Add tests for new features
- Update documentation
- Ensure cross-browser compatibility

## 🙏 Acknowledgments

- Built with Flask, scikit-learn, and modern web technologies
- Inspired by the need for accessible machine learning tools
- Thanks to the open-source ML community

---

**Made with ❤️ for democratizing machine learning**

*Transform your data into insights without writing a single line of code!*

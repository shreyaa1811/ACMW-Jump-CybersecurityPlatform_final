# 🔐 *Advanced AI-Powered Security Monitoring System*

This project is a comprehensive *Security Information and Event Management (SIEM)* system that integrates *Large Language Models (LLMs)* with modern security infrastructure to detect, analyze, and respond to security threats in real-time.

## 🚀 *Overview*

The system provides intelligent security monitoring by:

- Using *LLMs* to *generate realistic synthetic security logs* and events that mimic real-world threats
- Ingesting logs into a centralized database system for log management and analysis
- Enabling dynamic, *AI-assisted detection* of anomalies, threat patterns, and suspicious activities
- Offering *real-time dashboards* and visualizations for security analysts

This setup mimics the complexity of modern enterprise environments and is ideal for research, education, red teaming, or testing threat detection pipelines.

## 🏗 *System Architecture*

Our system consists of multiple integrated layers:

### 1. *Frontend Layer* - SecureVision Dashboard
- Interactive dashboard with multiple visualization tabs
- Real-time monitoring of security events
- Comprehensive threat analysis views

### 2. *AI Analytics API Service* 
- RESTful API endpoints for security data access
- ML model inference endpoints
- Data aggregation and analysis services

### 3. *Machine Learning Models*
- Intrusion Detection Model (XGBoost Classifier)
- RBA Anomaly Detection (Isolation Forest)
- Text Threat Detection (Random Forest Classifier)
- Synthetic Data Generator

### 4. *Data Sources*
- SSH Remote Server access to datasets
- Multiple security-related datasets
- Generated synthetic security events

### 5. *Database Layer*
- MySQL for Intrusion Detection & RBA data
- MongoDB for Text-based Threat data
- PostgreSQL for AI-enhanced events

### 6. *Data Processing Pipeline*
- Data loaders and ETL processes
- Feature engineering components
- Real-time monitoring modules

## 📋 *Prerequisites*

- Python 3.8+
- SSH access to remote data server
- Database access (MySQL, MongoDB, PostgreSQL)
- Sufficient disk space for datasets (especially RBA dataset ~8.5GB)
- Required Python packages (see requirements.txt)

## 🛠 *Installation*

1. *Clone the repository*
   ```
   bash
   git clone https://github.com/your-organization/security-monitoring-system.git
   cd security-monitoring-system
   ```

2. *Set up a virtual environment*
   ```
   bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. *Install dependencies*
   ```
   bash
   pip install -r requirements.txt
   ```

4. *Create environment file*
   
   Create a .env file in the project root directory with the following variables:
   ```
   # SSH Connection Settings
   SSH_HOST=your_ssh_host
   SSH_USER=your_ssh_username
   SSH_KEY_PATH=~/path/to/your/ssh_key.pem

   # Dataset paths on remote server
   INTRUSION_DETECTION_DATASET=path/to/cybersecurity_intrusion_data.csv
   AI_ENHANCED_DATASET=path/to/ai_ml_cybersecurity_dataset.csv
   TEXT_BASED_DATASET=path/to/cyber-threat-intelligence_all.csv
   RBA_DATASET=path/to/rba-dataset.csv

   # Database configurations
   MYSQL_HOST=your_mysql_host
   DEFAULT_MYSQL_USER=your_mysql_user
   DEFAULT_MYSQL_PASSWORD=your_mysql_password
   DEFAULT_MYSQL_DB=your_mysql_db

   POSTGRES_HOST=your_postgres_host
   DEFAULT_POSTGRES_USER=your_postgres_user
   DEFAULT_POSTGRES_PASSWORD=your_postgres_password
   DEFAULT_POSTGRES_DB=your_postgres_db

   MONGODB_HOST=your_mongodb_host
   DEFAULT_MONGODB_USER=your_mongodb_user
   DEFAULT_MONGODB_PASSWORD=your_mongodb_password
   DEFAULT_MONGODB_DB=your_mongodb_db
   ```

5. *Test your connection*
   ```
   bash
   python test_access.py
   ```

## 🚀 *Getting Started*

### Loading the Datasets

Load the security datasets into the databases:
```
bash
python database/database_loader.py
```

This process will:
- Connect to the remote SSH server
- Fetch the necessary datasets
- Load them into the appropriate databases
- Generate summary reports

### Starting the AI Analytics API

Start the API service:
```
bash
python run_analytics_service.py start
```

The API will be available at http://localhost:5000/

### Launching the Dashboard

Start the monitoring dashboard:
```
bash
python dash_app_main.py
```

The dashboard will be available at http://localhost:8050/

## 📊 *Datasets*

The system works with the following datasets:

1. *Intrusion Detection Dataset* - Network traffic and attack detection data
   - [Kaggle: Cybersecurity Intrusion Detection Dataset](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset)

2. *Text-based Cyber Threat Detection* - Security text analysis data
   - [Kaggle: Text-based Cyber Threat Detection](https://www.kaggle.com/datasets/ramoliyafenil/text-based-cyber-threat-detection)

3. *Risk-Based Authentication (RBA) Dataset* - Login and authentication data
   - [Kaggle: RBA Dataset](https://www.kaggle.com/datasets/dasgroup/rba-dataset)
   - Note: This is a large dataset (~8.5GB) with over 33M login records

## 👥 *Roles and Responsibilities*

### 1) *Security & Database Engineer*  
*Assigned to: Shreyaa*  
- *Focus*: Traditional security analysis and database handling  
- *Responsibilities*:  
  - Create rule-based detection systems using security patterns
  - Work with multiple data sources (MySQL, PostgreSQL, MongoDB)
  - Implement alert mechanisms for suspicious activities

### 2) *Data Infrastructure Engineer*  
*Assigned to: Sivangi*  
- *Focus*: Building the monitoring system and data pipelines
- *Responsibilities*:  
  - Set up data pipelines for processing security logs
  - Create dashboards for real-time attack monitoring
  - Implement data transformation and loading processes

### 3) *ML Engineer* 
*Assigned to: Darshitha*  
- *Focus*: Developing machine learning models for threat detection
- *Responsibilities*:  
  - Build anomaly detection systems
  - Identify unusual patterns in security data
  - Implement and optimize ML models for threat detection

### 4) *LLM Engineer* 
*Assigned to: Sandra*  
- *Focus*: Generating realistic security data using Large Language Models (LLMs)
- *Responsibilities*:  
  - Create diverse security scenarios for testing and training
  - Generate synthetic security events
  - Develop LLM-based threat analysis capabilities

## 🛠 *Component Details*

### AI Analytics API

The API provides the following endpoints:

- /health - Check API health status
- /api/security-data - Get synthetic security data
- /api/threat-analysis - Get threat analysis results
- /api/event-summary - Get statistical summary of events
- /api/events-by-time - Get events grouped by time
- /api/threat-intel - Get threat intelligence
- /api/all-analytics - Get all analytics data in one call
- /api/ml-prediction - Make ML model predictions

### Dashboard

The dashboard includes:

- *Overview Tab* - Global security posture and metrics
- *Intrusion Detection Tab* - Network traffic analysis and attack patterns
- *RBA Analysis Tab* - Login behavior and authentication risk analysis
- *Text Threats Tab* - Text-based threat intelligence
- *Real-time Monitoring Tab* - Live security alerts and system status

### Machine Learning Models

The system includes several ML models:

- *Intrusion Detection Model* - An XGBoost classifier that identifies network attacks
- *RBA Anomaly Detection* - An Isolation Forest model that detects unusual login behavior
- *Text Threat Detection* - A Random Forest classifier that analyzes text for security threats

## 🔍 *Usage Examples*

### Analyzing Security Events
```
python
from api_client import get_security_data, get_threat_analysis

# Get recent security events
events = get_security_data(hours=24, limit=100)

# Analyze threats
analysis = get_threat_analysis()
print(f"Detected {analysis['summary']['total_threats_detected']} threats")
```

### Monitoring for Anomalies
```
python
from api_client import detect_rba_anomalies

# Check for login anomalies
anomalies = detect_rba_anomalies(data)
high_risk = anomalies[anomalies['risk_category'] == 'High']
print(f"Found {len(high_risk)} high-risk login attempts")

```
### Running the Dashboard
```
bash
# Start the API service
python run_analytics_service.py start

# Start the dashboard
python dash_app_main.py --host=0.0.0.0 --port=8050
```

## 🐛 *Troubleshooting*

### Common Issues

1. *SSH Connection Problems*
   - Ensure your SSH key has correct permissions (chmod 600)
   - Verify the SSH server is accessible from your network

2. *Database Connection Issues*
   - Check that database credentials are correct in your .env file
   - Ensure database servers are running and accessible

3. *Large Dataset Handling*
   - For the RBA dataset, use chunked processing to avoid memory issues
   - Consider using a subset of data for testing

### Logging

Logs are available in:
- ai_analytics.log - API service logs
- dashboard.log - Dashboard application logs

## 🤝 *Team Members*

| Name      | Role                          | GitHub Handle                     |
|-----------|-------------------------------|-----------------------------------|
| Shreyaa   | Security & Database Engineer  | [@shreyaa1811](https://github.com/shreyaa1811)    |
| Sivangi   | Data Infrastructure Engineer  | [@essenbee227](https://github.com/essenbee227)    |
| Darshitha | ML Engineer                   | [@DarshithaG](https://github.com/DarshithaG)   |
| Sandra    | LLM Engineer                  | [@sandra-edathadan](https://github.com/sandra-edathadan)     |
| Prakash   | Project Lead                  | [@prakash-aryan](https://github.com/prakash-aryan)           |

## 🌐 *Real-World Applications*

This system has applications in various industries:

- *Financial Sector*: Fraud detection and compliance monitoring
- *Healthcare*: Patient data access monitoring and medical device security
- *Government*: Critical infrastructure protection
- *IoT*: Device behavior monitoring and anomaly detection

## 📄 *License*

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 *Acknowledgements*

- Thanks to Kaggle for providing the datasets
- Thanks to all team members for their contributions

---

*Happy Secure Monitoring!* 👩‍💻👨‍💻

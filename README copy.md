### To-Do 

- Test the file simple_api.py, you can run it with the command ```python run_analytics_service.py``` start and in another terminal run a few curl commands like : 
```
      curl -X POST http://localhost:5000/api/ml-prediction \
        -H "Content-Type: application/json" \
        -d '{
      "model_type": "rba",
      "data": {
        "login_frequency": 15,
        "location_change": 1,
        "device_change": 0,
        "time_since_last_login": 2.5,
        "failed_attempts": 3
      }
    }'
```


or 

```
    curl -X POST http://localhost:5000/api/ml-prediction \
        -H "Content-Type: application/json" \
        -d '{
      "model_type": "intrusion",
      "data": {
        "network_packet_size": 1500,
        "protocol_type": "TCP",
        "login_attempts": 5,
        "session_duration": 120,
        "encryption_used": "none",
        "ip_reputation_score": 0.3,
        "failed_logins": 3,
        "browser_type": "Chrome",
        "unusual_time_access": 1
      }
    }'
```


- For the dashboard use ``` python run_dashboard.py start ``` , see if the AI is working there and what needs to be fixed 

- A large language model (LLM) generates synthetic data from existing datasets. Machine learning models are trained to classify each data point as safe or unsafe. If unsafe, the type of attack is identified and the data point is stored accordingly in the database (**TODO**). All this information will be visualized on a dashboard.









Run Docker Compose
Now, we will run Docker Compose to set everything up.

Build and run the services:
bash
Copy
Edit
docker-compose up --build
Access the services:

Streamlit app: http://localhost:8501
Prometheus: http://localhost:9090
Grafana: http://localhost:3000 (use admin as the password)


Add Prometheus as a data source in Grafana:

Go to http://localhost:3000 and log in.
Add Prometheus as a data source by navigating to Configuration > Data Sources > Add Data Source.
Choose Prometheus and set the URL to http://prometheus:9090.
Create a Grafana Dashboard:

Create a simple dashboard in Grafana to visualize the model_accuracy metric.
You can use a Graph panel and set the metric to model_accuracy.
### Chatbot
After creating streamlit app locally

#### Local Docker 

1. build docker image
    ```bash
    docker build -t my-app .
    ```
2. Running docker
    ```bash
    docker run -p 8501:8501 my-app
    ```
3. Streamlit
    ```bash
    streamlit run deployment/app.py
    ```


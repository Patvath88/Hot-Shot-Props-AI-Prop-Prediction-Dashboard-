# Use Python 3.11 (compatible with xgboost + streamlit)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y build-essential && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config: disable browser prompt and set port
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]

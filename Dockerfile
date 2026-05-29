# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install standard system tools
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Add container health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set the command to run Streamlit
ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

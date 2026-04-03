# Use Python 3.10 slim — matches your Mac's Python version
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker caches this layer)
# If requirements don't change, Docker skips reinstalling packages
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project
COPY . .

# Create outputs directory inside container
RUN mkdir -p outputs

# Expose the port Gradio runs on
EXPOSE 7860

# Tell Gradio to listen on all interfaces (required for Docker)
ENV GRADIO_SERVER_NAME=0.0.0.0

# Start the app
CMD ["python3", "app.py"]
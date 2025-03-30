# Use a lightweight Python image as a base
FROM python:3.10-slim

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*  # Clean up after installation

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt to the container
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port 5000 to allow communication to and from the container
EXPOSE 5000

# Run the Flask app when the container starts
CMD ["python", "app.py"]

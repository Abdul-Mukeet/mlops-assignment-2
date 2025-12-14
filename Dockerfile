# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .
FROM apache/airflow:2.9.0
RUN pip install --no-cache-dir pandas scikit-learn
# Command to run when the container starts
CMD ["python", "src/train.py"]

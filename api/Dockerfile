FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install fastapi uvicorn scikit-learn pandas

# Copy code and model
COPY main.py .
COPY model.pkl .

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
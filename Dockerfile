# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY inference/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r inference/requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 3000
EXPOSE 3000

# Run the application
CMD ["python", "app.py"]
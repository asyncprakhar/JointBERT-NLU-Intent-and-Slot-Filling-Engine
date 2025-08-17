# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working current directory to /app
WORKDIR /app

# Copy the requirement file into the container
COPY requirements.txt .

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the application will run on
EXPOSE 8000

# Command to run application using unicorn
CMD ["uvicorn", "src.api.inference:app", "--host", "0.0.0.0", "--port", "8000"]
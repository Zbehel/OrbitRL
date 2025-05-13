# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy all project files into the container at /app
COPY . /app

# Run 'python main.py'
CMD ["python", "main.py"]
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY api/ ./api/

# Set the command to run the application
CMD ["gunicorn", "-b", "0.0.0.0:5000", "api.process_image:app"]

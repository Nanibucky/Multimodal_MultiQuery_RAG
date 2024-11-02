# Use the official slim Python image
# Use the official slim Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the required Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Install curl for health checks
RUN apt-get update && apt-get install -y curl

# Expose port 8080 (for Streamlit)
EXPOSE 7860

# Health check for the container to verify Streamlit is running
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run the Streamlit application
ENTRYPOINT ["streamlit", "run", "main_rag.py", "--server.port=7860", "--server.address=0.0.0.0"]

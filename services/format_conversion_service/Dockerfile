FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY services/format_conversion_service/requirements.txt ./service-requirements.txt
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt -r service-requirements.txt

# Copy service code
COPY services/format_conversion_service/ ./services/format_conversion_service/
COPY utilities/ ./utilities/

# Set environment variables
ENV PYTHONPATH="/app"
ENV PORT=8000
ENV HOST="0.0.0.0"
ENV CONTAINER_ENV="true"

# Expose the port
EXPOSE 8000

# Run the service
CMD ["uvicorn", "services.format_conversion_service.main:app", "--host", "0.0.0.0", "--port", "8000"]

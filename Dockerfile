# Use official Python image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . ./

# Set environment variables (optional defaults can be overridden by Railway)
# ENV TELEGRAM_BOT_TOKEN=""
# ENV TELEGRAM_CHAT_ID=""

# Command to run the agent; this could be adjusted as needed
CMD ["python", "linesniper_agent.py"]


# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app/ttyd

# Copy the current directory contents into the container at /usr/src/app
COPY . /app/ttyd

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
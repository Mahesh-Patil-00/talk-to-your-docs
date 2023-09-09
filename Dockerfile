# this dockerfile is meant as the base image for talk-to-your-data. We will install all the requirements.txt libraries in this base image.


# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app/ttyd

# Copy the current directory contents into the container at /usr/src/app
COPY requirements.txt /app/ttyd

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
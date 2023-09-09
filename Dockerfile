
# Use an official Python runtime as a parent image
FROM arslan2k12/ttyd_base

# Set the working directory in the container
WORKDIR /app/ttyd

# Copy the current directory contents into the container at /usr/src/app
COPY . /app/ttyd

# to make gradio app accessible to local network (default 127.0.0.1 is only accissible within the container)
ENV GRADIO_SERVER_NAME=0.0.0.0

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Use ENTRYPOINT to allow passing user arguments
ENTRYPOINT ["python", "app.py"]
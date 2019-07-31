# Python base image
FROM python:3.7.3

# Where the API server lives
WORKDIR /app/

# Install required dependencies
COPY requirements.txt /app/
RUN pip install -r ./requirements.txt

# Copy data file and pickle file
COPY anime.csv my_dt_model.pkl /app/

# API server
COPY api.py /app/


# Container port on which the server will be listening
#EXPOSE 7777


# Launch server app
ENTRYPOINT python ./api.py
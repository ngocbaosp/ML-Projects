# Python base image
FROM python:3.7.3

# Where the API server lives
WORKDIR /app/


# Install required dependencies
RUN pip install requests
RUN pip install wtforms
RUN pip install flask_wtf


# App folder
COPY . /app/


# Container port on which the server will be listening
#EXPOSE 5000


# Launch web app
ENTRYPOINT python ./app.py
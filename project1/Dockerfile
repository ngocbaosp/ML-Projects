# Python base image
FROM python:3.7.3

# Where the API server lives
WORKDIR /app/

# Install required dependencies
#COPY requirements.txt /app/
#RUN pip install -r ./requirements.txt
RUN pip install requests
RUN pip install wtforms
RUN pip install flask_wtf

# App folder
COPY . /app/
# Container port on which the server will be listening
#EXPOSE 5000
# Launch web app
ENTRYPOINT python ./app.py
#CMD ["flask", "run", "--host", "0.0.0.0"]
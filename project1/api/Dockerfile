# Python base image
FROM python:3.7.3

# Where the API server lives
WORKDIR /app/

# Install required dependencies
COPY requirements.txt anime.csv my_dt_model.pkl /app/
RUN pip install -r ./requirements.txt

COPY anime.csv my_dt_model.pkl /app/
# API server
COPY api.py /app/
# Container port on which the server will be listening
#EXPOSE 7777
# Launch server app
ENTRYPOINT python ./api.py
#CMD ["flask", "run", "--host", "0.0.0.0"]
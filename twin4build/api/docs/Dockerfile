FROM ubuntu:latest 

# set a directory for the app
WORKDIR /

# install dependencies
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install unzip -y
RUN apt-get install python3.10 -y
RUN apt-get install pip -y
RUN apt-get install graphviz -y
RUN pip install psycopg2-binary
COPY . .
RUN unzip Twin4build_code_working.zip
WORKDIR /Twin4build_code/Twin4Build
RUN pip install -r twin4build_req.txt

# define the port number the container should expose
#EXPOSE 8005

# run the command
#CMD ["python3", "twin4build/api/codes/ml_layer/simulator_api.py"]
#CMD ["python3", "twin4build/api/codes/ml_layer/request_to_api.py"]

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]

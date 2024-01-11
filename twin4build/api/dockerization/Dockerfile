FROM ubuntu:latest

# set a directory for the app
WORKDIR /

# install dependencies
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install unzip -y
RUN apt-get install python3.11 -y
RUN apt-get install pip -y
RUN pip install typing-extensions
RUN apt-get install graphviz -y
RUN pip install psycopg2-binary
RUN apt-get install git -y
#Installing torch for cpu to avoid CUDA libraries problems, would need to refactor to use CUDA
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu



RUN git clone https://github.com/JBjoernskov/Twin4Build.git

WORKDIR /Twin4Build



#For DEVELOPMENT branches, change the branch to twin4build_api_updates
#Comment if you want to use the main branch
#RUN git checkout twin4build_api_updates

#Install the dependencies
RUN pip install -r requirements_linux_no_version.txt

#Expose the 8070 port8
EXPOSE 8070

# copy docker container startup script  
COPY ./docker-entrypoint.sh /docker-entrypoint.sh
#Make it executable
RUN chmod +x /docker-entrypoint.sh
#Set the entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]

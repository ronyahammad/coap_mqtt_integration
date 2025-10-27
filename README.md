# coap_mqtt_integration
How to run the whole project?

At first download the folder from github.

Then Install Docker desktop. After installing the Docker, open vscode terminal and go to project directory and type docker-compose up -d --build

it will install all the libraries and python version required. When it is installed, you can go to docker desktop and go to the container named coap_mqtt_integration (suppose you kept the folder name same as github folder name).

should check the containers, specially mqtt_subsciber container where the data from all pipelines can be seen.

How to configure influx BD and Grafana for data store and visualization?

For influx BD, open http://localhost:8086/signin in the broswer and in the username type admin and password is admin123


after login go to the Load Data and then go to the sources>influx CLI>initialize client, copy --token string and paste it into .env file's INFLUXDB_INIT_ADMIN_TOKEN. after that in the vscode terminal at first type docker-compose down. when it finishes, type again docker-compose up -d --build. then go to the http://localhost:8086/signin browser and put the same username and password and then go to the 
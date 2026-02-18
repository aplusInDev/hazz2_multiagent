# ejabberdctl register master ejabberd master_pass
# ejabberdctl register qagent ejabberd qagent_pass
# ejabberdctl register randomagent ejabberd random_pass
# ejabberdctl register human ejabberd human_pass

docker-compose up -d ejabberd
docker exec -it ejabberd ejabberdctl register master ejabberd master_pass
docker exec -it ejabberd ejabberdctl register qagent ejabberd qagent_pass
docker exec -it ejabberd ejabberdctl register randomagent ejabberd random_pass
docker exec -it ejabberd ejabberdctl register human ejabberd human_pass
docker-compose up -d master_agent
docker-compose up -d qlearning_agent random_agent human_agent --build
#!/usr/bin/env bash
set -e

echo "> Starting ejabberd..."
docker compose up -d ejabberd

echo "> Waiting for ejabberd to be ready..."
until docker exec ejabberd ejabberdctl status >/dev/null 2>&1; do
  sleep 2
done

echo "> ejabberd is ready"

echo "> Registering users..."
docker exec ejabberd ejabberdctl register master ejabberd master_pass || true
docker exec ejabberd ejabberdctl register qagent ejabberd qagent_pass || true
docker exec ejabberd ejabberdctl register randomagent ejabberd random_pass || true
docker exec ejabberd ejabberdctl register human ejabberd human_pass || true
docker exec ejabberd ejabberdctl register heuristic ejabberd heuristic_pass || true

echo "> Users registered (existing users ignored)"

echo "> Starting master agent..."
docker compose up -d --build master_agent

echo "> Starting other agents..."
docker compose up -d --build qlearning_agent random_agent human_agent heuristic_agent

echo "> All services are up"

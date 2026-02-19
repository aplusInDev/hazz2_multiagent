<#
.SYNOPSIS
Run the multi-agent system (ejabberd + agents)

.DESCRIPTION
- Starts ejabberd
- Waits until it is fully ready
- Registers XMPP users
- Builds and runs all agents
#>


# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Stop"

Write-Host "> Starting ejabberd..."
docker compose up -d ejabberd

Write-Host "> Waiting for ejabberd to be ready..."
do {
    Start-Sleep -Seconds 2
    $status = docker exec ejabberd ejabberdctl status 2>$null
} until ($status -match "is running")

Write-Host "> ejabberd is ready"

Write-Host "> Registering users..."
docker exec ejabberd ejabberdctl register master ejabberd master_pass 2>$null
docker exec ejabberd ejabberdctl register qagent ejabberd qagent_pass 2>$null
docker exec ejabberd ejabberdctl register randomagent ejabberd random_pass 2>$null
docker exec ejabberd ejabberdctl register human ejabberd human_pass 2>$null
docker exec ejabberd ejabberdctl register heuristic ejabberd heuristic_pass 2>$null

Write-Host "> Users registered (existing users ignored)"

Write-Host "> Starting master agent..."
docker compose up -d --build master_agent

Write-Host "> Starting other agents..."
docker compose up -d --build qlearning_agent random_agent human_agent heuristic_agent

Write-Host "> All services are up"

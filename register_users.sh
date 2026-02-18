#!/bin/bash
set -e

EJABBERD_HOST="${EJABBERD_HOST:-localhost}"
EJABBERD_CONTAINER="${EJABBERD_CONTAINER:-ejabberd}"

echo "Waiting for ejabberd to be ready..."
until docker exec "$EJABBERD_CONTAINER" ejabberdctl status 2>/dev/null | grep -q "is running"; do
  sleep 2
done
echo "ejabberd is ready."

register_user() {
  local user=$1
  local password=$2
  echo "Registering $user@ejabberd..."
  docker exec "$EJABBERD_CONTAINER" ejabberdctl register "$user" ejabberd "$password" 2>/dev/null || \
    echo "  (already registered or error - continuing)"
}

register_user master master_pass
register_user qagent qagent_pass
register_user randomagent random_pass
register_user human human_pass

echo "All users registered."

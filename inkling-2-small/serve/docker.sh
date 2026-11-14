# Docker on this host. tomcupr is not in the docker group, so the default
# is sudo. A failed inspect is a failure, not "container absent".
: "${DOCKER:=sudo docker}"

_docker() {
  # DOCKER is a command, possibly "sudo docker".
  # shellcheck disable=SC2086
  $DOCKER "$@"
}

# Prints "absent" or "present" and returns 0. Any other docker result
# returns 1 after writing the docker output.
container_state() {
  local name=$1 out rc
  set +e
  out=$(_docker inspect --format '{{.Name}}' "$name" 2>&1)
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    echo present
    return 0
  fi
  if printf '%s\n' "$out" | grep -Eqi 'no such (object|container)'; then
    echo absent
    return 0
  fi
  printf '%s\n' "$out" >&2
  echo "docker inspect ${name} failed (exit ${rc}). That is not treated as absent." >&2
  return 1
}

require_absent() {
  local name=$1 state
  state=$(container_state "$name")
  if [ "$state" != absent ]; then
    echo "container ${name} is present. This script will not stop it." >&2
    return 1
  fi
}

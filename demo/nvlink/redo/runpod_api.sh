#!/usr/bin/env bash
# Shared RunPod API helpers (REST rest.runpod.io/v1 + GraphQL api.runpod.io).
# Source this file. Requires: curl, jq, ~/.runpod_api_key.
set -u

RP_KEY="$(cat "$HOME/.runpod_api_key")"
RP_REST="https://rest.runpod.io/v1"
RP_GQL="https://api.runpod.io/graphql"

rp_rest() { # rp_rest METHOD PATH [JSON_BODY]
  local method="$1" path="$2" body="${3:-}"
  if [ -n "$body" ]; then
    curl -sS --max-time 60 -X "$method" "$RP_REST$path" \
      -H "Authorization: Bearer $RP_KEY" -H "Content-Type: application/json" \
      -d "$body"
  else
    curl -sS --max-time 60 -X "$method" "$RP_REST$path" \
      -H "Authorization: Bearer $RP_KEY"
  fi
}

rp_balance() { # prints the account clientBalance (USD, may be negative)
  curl -sS --max-time 60 "$RP_GQL" \
    -H "Content-Type: application/json" -H "Authorization: Bearer $RP_KEY" \
    -d '{"query":"query { myself { clientBalance } }"}' \
    | jq -r '.data.myself.clientBalance'
}

rp_pod() { # rp_pod POD_ID -> pod JSON
  rp_rest GET "/pods/$1"
}

rp_terminate() { # rp_terminate POD_ID -> delete (terminate) the pod
  rp_rest DELETE "/pods/$1"
}

rp_ssh_dest() { # rp_ssh_dest POD_ID -> "IP PORT" of the public SSH mapping
  rp_pod "$1" | jq -r '
    (.publicIp // empty) as $ip
    | (.portMappings["22"] // empty) as $port
    | if $ip != "" and $port != null then "\($ip) \($port)" else empty end'
}

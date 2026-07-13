#!/usr/bin/env bash
set -euo pipefail

for job_id in "$@"; do
  tsp -w "$job_id"
done

#!/bin/bash
cd "$(dirname "$0")"
while IFS= read -r q; do
  [ -z "$q" ] && continue
  echo "=== $q"
  python axq.py "$q" 25 submittedDate 400
  sleep 25
done < queries.txt

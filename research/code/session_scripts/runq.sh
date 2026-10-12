#!/bin/bash
cd "$(dirname "$0")"
while IFS= read -r q; do
  echo "=== $q"
  python axs.py "$q" 50 | cut -c1-700
  sleep 30
done < "$1"

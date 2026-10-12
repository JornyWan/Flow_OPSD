#!/bin/bash
cd "$(dirname "$0")"
while IFS= read -r q; do
  [ -z "$q" ] && continue
  echo "##### $q"
  python axs.py "$q" 50 300
  sleep 20
done < axq_list.txt
echo DONE

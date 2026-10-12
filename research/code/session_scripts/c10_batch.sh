D=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
while IFS= read -r q; do
  [ -z "$q" ] && continue
  python3 $D/c10_hf.py "$q" 12 >> $D/c10_out.txt 2>&1
  echo >> $D/c10_out.txt
  sleep 15
done < $D/c10_queries.txt
echo DONE >> $D/c10_out.txt

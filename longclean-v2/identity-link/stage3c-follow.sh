until ! tmux has-session -t lc-counter-refine 2>/dev/null; do sleep 3; done
idle() {
  for _ in $(seq 1 12); do
    if test "$(nvidia-smi -i 3 --query-gpu=utilization.gpu --format=csv,noheader,nounits)" = 0 &&
       test -z "$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv,noheader,nounits)"; then return 0; fi
    sleep 3
  done
  return 75
}
for label in 6 10 14 low; do
  name=counter-dev-rim-$label
  if test -f "$name.exit" && test "$(cat "$name.exit")" = 75; then
    inset=$label; clearance=9
    if test "$label" = low; then inset=10; clearance=4; fi
    idle && bash stage3c-run.sh counters 3 "$name-retry" --phase dev --cases rim --rim-inward-mm "$inset" --rim-clearance-mm "$clearance" > "$name-retry.log" 2>&1
  fi
done
idle && bash stage3c-run.sh counters 3 counter-unknown-3 --phase blind --cases identity_unknown --unknown-sauce > counter-unknown-3.log 2>&1
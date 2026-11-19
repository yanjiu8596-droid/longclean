card=$1
for _ in $(seq 1 90); do
  if test -s "rim-profiled-slow-$card.exit" &&
     test "$(nvidia-smi -i "$card" --query-gpu=utilization.gpu --format=csv,noheader,nounits)" = 0 &&
     test -z "$(nvidia-smi -i "$card" --query-compute-apps=pid --format=csv,noheader,nounits)"; then
    exec bash stage3d-pi.sh natural "$card" "stage3d-natural-$card"
  fi
  sleep 3
done
exit 75

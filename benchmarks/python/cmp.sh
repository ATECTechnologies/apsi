# Run from this folder after: for f in nbody objs strs; do apsi convert $f.py; done
for f in nbody objs strs; do
  p=99; a=99
  for i in 1 2 3 4 5; do
    x=$(python3 $f.py | awk '{print $NF}' | tr -d s); y=$(apsi run $f.apsi | awk '{print $NF}' | tr -d s)
    p=$(echo "if ($x<$p) $x else $p" | bc); a=$(echo "if ($y<$a) $y else $a" | bc)
  done
  echo "$f python $p apsi $a"
done

for i in {00..77}; do
    if test -f shard_$i.csv; then
        tail -n +2 -q shard_$i.csv > tmp.csv && mv tmp.csv shard_$i.csv
    fi
done
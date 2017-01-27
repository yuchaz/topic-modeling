
option="${1}"

if (( option == -m)); then
    echo "Preprocessing..."
    python main.py;
fi

if (( option == -m | option == -c)); then
    echo "Training..."
    python classify.py;
fi
if (( option == -m | option == -c | option == -v)); then
    echo "Evaluation..."
    python visualize.py;
fi

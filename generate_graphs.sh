#!/bin/bash

# Create dependency graphs directory
mkdir -p dependency_graphs

# List of main Python files
MAIN_FILES=("app.py" "bot.py" "enhanced_player.py" "game_constants.py" "hokm.py" "run_game.py" "test_server.py" "train_backend.py" "train_hokm.py")

# Loop through files and generate dependency graphs
for file in "${MAIN_FILES[@]}"; do
    echo "Generating dependency graph for $file..."
    
    # Generate PDF graph
    pydeps "$file" --max-bacon=10 --cluster --rankdir=LR -T pdf -o "dependency_graphs/${file%.py}_deps.pdf"
    
    # Generate SVG graph (more interactive)
    pydeps "$file" --max-bacon=10 --cluster --rankdir=LR -T svg -o "dependency_graphs/${file%.py}_deps.svg"
    
    # Generate a non-clustered version for more detailed view
    pydeps "$file" --max-bacon=10 --rankdir=LR -T svg -o "dependency_graphs/${file%.py}_detailed_deps.svg"
    
    echo "Done with $file"
    echo "-----------------------------------"
done

echo "All dependency graphs generated in the 'dependency_graphs' directory."
echo "PDF files are good for printing, SVG files are better for interactive viewing." 
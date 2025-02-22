import argparse
import matplotlib.pyplot as plt
import re
from pathlib import Path
from typing import Dict, List

def parse_opts() -> Dict[str, bool]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing_data_path", type=str, default='./images/profiling_21_02/unoptimised_starter/timing_data.txt', help="Path to the timing data file")
    args = parser.parse_args()
    return {"timing_data_path": Path(args.timing_data_path)}

def read_timing_data(file_path: Path) -> Dict[str, List[float]]:
    """Read timing data from the file and extract values."""
    timing_data = {}
    current_operation = None
    
    print(f"Reading timing data from {file_path}")
    
    with open(str(file_path / 'timing_data.txt'), 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # Check if this is an operation name line
        if line.strip().endswith(':'):
            current_operation = line.strip()[:-1]
            timing_data[current_operation] = []
        # Extract numeric values
        elif line.strip().startswith(('Average:', 'Last:', 'Min:', 'Max:')):
            value_match = re.search(r':\s*([\d.]+)', line)
            if value_match:
                value = float(value_match.group(1))
                timing_data[current_operation].append(value)
    
    return timing_data

def create_box_plot(timing_data_path: Path, timing_data: Dict[str, List[float]]) -> None:
    """Create and save a box plot of the timing data."""
    plt.figure(figsize=(12, 12))
    
    # Prepare data for plotting
    labels = []
    data = []
    
    for operation, values in timing_data.items():
        if operation != "":  # Skip empty operation names
            labels.append(operation)
            data.append(values)
    
    # Create box plot
    plt.boxplot(data, labels=labels, vert=True)
    
    # Customize the plot
    plt.title('Operation Timing Distribution')
    plt.ylabel('Time (s)')
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{timing_data_path}/timing_boxplot.png')
    plt.close()

def main(timing_data_path: Path):
    # Read timing data
    timing_data = read_timing_data(timing_data_path)
    
    # Remove the largest yolo_prediction value
    # timing_data["yolo_prediction"].pop(0)
    average_time = 0
    for operation, values in timing_data.items():
        average_time += values[0]
  
    print(f"Average time to process a single (successful) frame (s): {average_time / 1_000_000_000}")
    
    # Create and save the box plot
    create_box_plot(timing_data_path, timing_data)

if __name__ == "__main__":
    opts = parse_opts()
    main(opts["timing_data_path"])

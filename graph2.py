import matplotlib.pyplot as plt
import pandas as pd
import sys

def plot_stacked_graph(title, output, csv_file1, csv_file2, name1, name2, position, position2):
    # Read CSV files
    data1 = pd.read_csv(csv_file1, header=0)
    data2 = pd.read_csv(csv_file2, header=0)
    
    # Extract column titles
    x1_label, y1_label = data1.columns[0], data1.columns[1]
    x2_label, y2_label = data2.columns[0], data2.columns[1]

    # Extract data
    x1, y1 = data1[x1_label], data1[y1_label]
    x2, y2 = data2[x2_label], data2[y2_label]

    # Create the plot
    fig, ax = plt.subplots()

    line1, = ax.plot(x1, y1, label=f"{name1}")
    line2, = ax.plot(x2, y2, label=f"{name2}")

    if position < 2 ** 32:
        ax.axvline(x=position, color='red', linestyle='--')
        ax.text(position, ax.get_ylim()[1] * 0.95, f"Feed-forward average\n # of epochs ({position})", color=line2.get_color(), fontsize=10, ha='right', va='top', backgroundcolor='white')
    if position2 < 2 ** 32:
        ax.axvline(x=position2, color='red', linestyle='--')
        ax.text(position2, ax.get_ylim()[1] * 0.75, f"Deep learning average\n # of epochs ({position2})", color=line1.get_color(), fontsize=10, ha='right', va='top', backgroundcolor='white')

    ax.set_xlabel(x1_label)
    ax.set_ylabel(y1_label)
    ax.legend()
    ax.set_title(title, fontsize=12)

    plt.savefig(output)

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: python script.py <title> <output_file> <csv_file1> <csv_file2> <csv1_name> <csv2_name> <pos1> <pos2>")
        sys.exit(1)

    csv_file1 = sys.argv[3]
    csv_file2 = sys.argv[4]
    title = sys.argv[1]
    output = sys.argv[2]
    position = sys.argv[5]
    position2 = sys.argv[6]

    plot_stacked_graph(title, output, csv_file1, csv_file2, position, position2, int(sys.argv[7]), int(sys.argv[8]))

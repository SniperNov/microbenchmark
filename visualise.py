import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import yaml

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def extract_parameters(content):
    pattern = r"(target[^]+?) \[threads=(\d+) teams=(\d+)\]\n((?:\t[\d.]+[±±][\d.]+)+)"
    return re.findall(pattern, content)

def parse_data(matches, Ns):
    results = []
    for method, threads, teams, data_line in matches:
        entries = re.findall(r"([\d.]+)[±±]([\d.]+)", data_line)
        for i, (avg, err) in enumerate(entries):
            results.append({
                'method': method.strip(),
                'threads': int(threads),
                'teams': int(teams),
                'N': Ns[i] if i < len(Ns) else -1,
                'avg': float(avg),
                'err': float(err)
            })
    return results

def plot_grouped(results, group_by, xaxis, outdir):
    grouped = defaultdict(list)
    for item in results:
        label = item[group_by]
        grouped[label].append((item[xaxis], item['avg'], item['err']))

    for label, series in grouped.items():
        series.sort()
        x, y, err = zip(*series)
        plt.figure()
        plt.errorbar(x, y, yerr=err, fmt='-o', capsize=5)
        plt.title(f"{label}: {xaxis} vs Overhead")
        plt.xlabel(xaxis)
        plt.ylabel("Offloading Overhead (μs)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        filename = f"{xaxis}_vs_overhead_{label.replace(' ', '_').replace(':','').replace('(','').replace(')','')}.png"
        plt.savefig(os.path.join(outdir, filename))
        plt.close()

def load_yaml_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Smart visualisation of OpenMP benchmark results')
    parser.add_argument('filename', type=str, help='Output file to parse')
    parser.add_argument('--job', type=str, default='job')
    parser.add_argument('--machine', type=str, default='machine')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    args = parser.parse_args()

    config = load_yaml_config(args.config) if args.config else {}
    generate = config.get('generate_plots', {
        'n_vs_overhead': True,
        'thread_vs_overhead': True,
        'team_vs_overhead': True
    })

    content = read_file(args.filename)
    n_line = re.search(r"Method/N\s+(.+)", content)
    Ns = list(map(int, n_line.group(1).split())) if n_line else list(range(14))

    matches = extract_parameters(content)
    if not matches:
        print("No matches found in output. Check output format.")
        return

    results = parse_data(matches, Ns)
    outdir = os.path.join("Plots", args.machine, args.job)
    os.makedirs(outdir, exist_ok=True)

    if generate.get('n_vs_overhead', False):
        plot_grouped(results, group_by='method', xaxis='N', outdir=outdir)
    if generate.get('thread_vs_overhead', False):
        plot_grouped(results, group_by='method', xaxis='threads', outdir=outdir)
    if generate.get('team_vs_overhead', False):
        plot_grouped(results, group_by='method', xaxis='teams', outdir=outdir)

    print(f"Plots saved to {outdir}")

if __name__ == "__main__":
    main()

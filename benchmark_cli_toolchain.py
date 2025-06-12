#!/usr/bin/env python3

import argparse
import subprocess
import os
import json
import yaml
from datetime import datetime

# ---- CONFIG PARSER ----
def load_config(config_file):
    with open(config_file, 'r') as f:
        if config_file.endswith(".yaml") or config_file.endswith(".yml"):
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError("YAML config is empty or invalid.")
            return config
        elif config_file.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError("Unsupported config file format. Use .yaml or .json")

# ---- BUILD STEP ----
def build_microbenchmark():
    print("[BUILD] Compiling microbenchmark using Makefile...")
    subprocess.run(["make", "clean"], check=True)
    subprocess.run(["make"], check=True)

# ---- RUN STEP ----
def run_microbenchmark(methods, Ns, thread_counts, team_counts, outfile):
    print("[RUN] Launching benchmark...")
    method_str = "Method=" + ",".join(map(str, methods))
    n_str = "N=" + ",".join(map(str, Ns))
    thread_str = "thread_count=" + ",".join(map(str, thread_counts))
    team_str = "team_count=" + ",".join(map(str, team_counts))

    cmd = ["./microbenchmark", method_str, n_str, thread_str, team_str]
    print("[RUN] Command:", " ".join(cmd))

    with open(outfile, 'w') as f:
        subprocess.run(cmd, stdout=f, check=True)

# ---- VISUALISE STEP ----
def run_visualisation(method, filename, job, IDA, machine, config_file):
    print("[PLOT] Generating analysis and plots...")
    subprocess.run([
        "python3", "visualise.py",
        filename,
        "--job", job,
        "--machine", machine,
        "--config", config_file
    ], check=True)

# ---- MAIN ENTRY ----
def main():
    parser = argparse.ArgumentParser(description="OpenMP Microbenchmark Automation CLI")
    parser.add_argument('--config', type=str, help='Path to YAML or JSON config file')
    args = parser.parse_args()

    if not args.config:
        print("Please specify a configuration file using --config")
        return

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return

    # Extract config values
    methods = config.get('methods', list(range(1, 13)))
    Ns = config.get('Ns', [1024])
    thread_counts = config.get('threads', [32])
    team_counts = config.get('teams', [64])
    job = config.get('job', 'default_job')
    IDA = config.get('IDA', 27)
    machine = config.get('machine', 'unknown_machine')
    method_tag = config.get('method_tag', 'arraybench')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"output_{job}_{timestamp}.out"

    build_microbenchmark()
    run_microbenchmark(methods, Ns, thread_counts, team_counts, output_file)
    run_visualisation(method_tag, output_file, job, IDA, machine, args.config)

    print(f"[DONE] Benchmark complete. Output saved to: {output_file}")

if __name__ == "__main__":
    main()

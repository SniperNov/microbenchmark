### **📜 README: OpenMP Offloading Microbenchmark**  

# **OpenMP Offloading Microbenchmark**  
This repository contains a **microbenchmark suite** designed to evaluate OpenMP offloading performance across different architectures and compilers. The benchmark measures **offloading overhead, execution time scaling, and OpenMP target device behavior**.

---

## **📂 Repository Contents**  

| **File** | **Description** |
|----------|---------------|
| `microbenchmark.c` | Main benchmark source code, implementing multiple OpenMP offloading strategies. |
| `common.c` | Common utility functions used by the benchmark. |
| `common.h` | Header file defining shared functions and data structures. |
| `Makefile` | Build system to compile the benchmark with different compilers. |
| `Makefile.defs.gcc` | Compiler-specific definitions for GCC-based compilation. |

---

## **📥 Installation & Compilation**  

### **1️⃣ Clone the Repository**
To obtain the latest version of the benchmark:
```bash
git clone https://github.com/yourusername/microbenchmark.git
cd microbenchmark
```

### **2️⃣ Load Required Modules (Depending on Target Machine)**
#### **🔹 For example using cray for AMD MI210 on ARCHER2**
```bash
module load PrgEnv-cray
module load rocm
```

#### **🔹 For GCC on Cirrus**
```bash
module load gcc/12.3.0
```

### **3️⃣ Compile the Benchmark**
- **For default execution offload to GPU**:
  ```bash
  make
  ```
- **For OpenMP offloading overhead distribution analysis**:
  ```bash
  make microbenchmark_distribution
  ```
---

## **🛠 Running the Benchmark**
### **1️⃣ Default Execution**
```bash
srun microbenchmark
```
Runs all OpenMP offloading strategies and outputs a **performance table**.

### **2️⃣ Specifying Methods and Data Size**
```bash
srun ./microbenchmark 1 2 3 4 1024
```
Runs **methods 1, 2, 3, and 4** with **array size 1024**.

### **3️⃣ Running All Benchmarks and Saving Output**
```bash
make run_all
```
Runs all methods and saves results to `results.txt`.

### **4️⃣ Generating Offloading Overhead Distribution Plot**
```bash
make run_plot
```
- Executes `microbenchmark_distribution` with **distribution printing enabled**.
- Saves **overhead values** to `overhead_distribution.txt`.
  
---

## **📝 Most Recent Modifications**
### **1️⃣ Changing Compiler Options**
Modify `Makefile` to switch different offloading method
- **5:** `target parallel`
- **9:** `nowait`

### **2️⃣ Modifying OpenMP Methods**
Edit `microbenchmark.c`:
- Modify `device_target()` to add/remove offloading techniques.
- Adjust OpenMP `#pragma` directives for different memory mappings.

### **3️⃣ Adding New Features**
1. Implement a new warmup method in `microbenchmark.c`.
2. Add new compilation options in `Makefile`.
3. Submit a pull request via GitHub.

---

## **📧 Support & Contact**
For issues or improvements, please **open an issue** on GitHub or contact:  
📧 **tuweiyu7749@gmail.com**  

🚀 **Happy benchmarking!** 😊

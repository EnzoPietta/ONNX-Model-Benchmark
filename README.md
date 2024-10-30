# ONNX-Model-Benchmark

### Python Dependencies

- `pandas`
- `seaborn`
- `onnx`
- `onnxruntime`
- `onnxruntime-gpu` 
- `tqdm`
- `pyJoules`
- `torch`

#### Notes:

- **Dataset:**  
  The dataset used in this project can be downloaded from the following link:  
  [Download Dataset](https://drive.google.com/drive/folders/1QN67qTdgyFAbELnOuU3UREMoXhyl7OHg?usp=sharing)

- **GPU Inference:**  
  - Check CUDA compatibility on the official site: [CUDA Execution Provider Requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements).
  - To measure power consumption when using the GPU, you need to install the **NVIDIA Management Library (NVML)**.

- **Power Measurement on CPU (Linux):**  
  To measure CPU power consumption on Linux, you must set proper permissions by running the following command:
  ```bash
  sudo chmod -R a+r /sys/class/powercap/intel-rapl
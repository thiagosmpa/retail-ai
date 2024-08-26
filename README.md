## RETAIL AI - Intelligent Analysis for Retail

**Enhance the shopping experience and optimize your operations with the power of Computer Vision.**

![RETAIL AI](src/output.gif)

This project leverages advanced AI technologies to provide valuable insights into customer behavior in retail environments, such as foot traffic, dwell time, and even demographic estimates.

### Key Features:

* **Person Detection and Tracking:** Identifies and tracks the movement of individuals in real-time.
* **Gender and Age Estimation:** Estimates demographic information of customers based on facial detection.
* **Dwell Time Analysis:** Calculates the time each person spends in different areas of the store.
* **Heatmaps:** Generates heatmaps to visualize customer flow and identify high-interest zones.
* **Detailed Logs:** Records crucial information in JSON and CSV formats for later analysis.
* **Entry and Exit Counting:** Monitors the number of people entering and leaving specific areas.

### YOLO Model Execution:

The `run_yolo_models` function simplifies the execution of YOLO models for detection, tracking, and pose estimation.

```python
def run_yolo_models(model_path, task, format, **kwargs):
    """
    Automatically runs YOLO models for detection, tracking, or pose estimation.

    Args:
        model_path (str): Path to the model file.
        task (str): Task to be performed by the model ('track', 'detect', or 'pose').
        format (str): Format of the model file ('openvino', 'onnx', or 'pt').
        **kwargs: If "classes" is passed, the model will filter detections by class.
    """
```

### Project Structure:

* **`main.py`**: Responsible for running the full project inference on videos, applying all functionalities together to generate comprehensive analyses.
* **`visualize.ipynb`**: Jupyter Notebook that demonstrates the project's individual techniques (detection, tracking, age and gender estimation, etc.) in isolation on static images, facilitating the understanding of each step of the process.

### Downloads and Repositories:

* **Project Files (including original video used in inference and related outputs):** [Retail files](https://drive.google.com/drive/folders/1XzXzfcilRSrZhu5I0jb4mRgxC1q4WJiP?usp=share_link)
* **Mivolo Weights and Checkpoints:** [Mivolo weights / Mivolo checkpoint](https://drive.google.com/drive/folders/1FagDwoq8GfayuBLEye5IolINvF-9ixDO?usp=share_link)
* **Mivolo Repository:** [Mivolo model link](https://github.com/WildChlamydia/MiVOLO)

### Next Steps:

* **Installation:** Details on how to set up the environment and install the necessary dependencies.
* **Execution:** Step-by-step instructions to run the project, including examples of how to use the `run_yolo_models` function and how to utilize `main.py` and `visualize.ipynb`.
* **Configuration:** Explanation on how to customize the project for different scenarios and needs.
* **Contribution:** Guidelines on how to contribute to the project’s development.

**With RETAIL AI, transform data into strategic decisions for your business.**
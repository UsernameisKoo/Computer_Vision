

# Visualization/Plot of NTU-RGB Skeleton with Python    

This project provides tools to visualize the skeleton data from the NTU_RGB+D dataset using Python. While the dataset authors provided MATLAB code for visualization, this repository offers an alternative Python-based implementation. Existing online solutions were either non-functional or lacked proper documentation, prompting the development of this repository.

## Installation
To use this repository, follow these steps:

1. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

   
2. Download and install FFmpeg to enable video saving. It is easy to install and works reliably across platforms. You can find it [here](https://ffmpeg.org/download.html)


## Usage
Refer to `npy_skeleton_analyzer.ipynb` for a quick introduction. All functions are documented for ease of use.  

Videos are stored in the `videos/` folder.

### Example Output:
Pre-generated examples are available in the `videos/` folder. Below is a sample expected output:

![Skeleton Animation](videos/gif.gif)

## Acknowledgments
This project was adapted and expanded from the original repository  
[gardiens/plot_skeleton_NTU_RGB-D](https://github.com/gardiens/plot_skeleton_NTU_RGB-D)  
under the **MIT License**.  
We appreciate the author’s work for making Python-based NTU-RGB+D visualization accessible.

Special thanks to the [Rose Lab](https://rose1.ntu.edu.sg/dataset/actionRecognition/) for providing the NTU_RGB+D dataset, which enables research on this challenging problem.

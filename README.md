# Football AI - Advanced Sports Analytics with Computer Vision

A comprehensive Football AI system that leverages computer vision and machine learning to analyze football/soccer matches in real-time. This system detects players, tracks their movements, classifies teams, and provides detailed match analytics with pitch-level visualizations.

## 🎯 Features

### Player & Object Detection
- **Ball Detection**: Accurately identifies the ball in video frames
- **Player Detection**: Detects all players on the field with high precision
- **Goalkeeper Detection**: Identifies goalkeepers separately from regular players
- **Referee Detection**: Tracks referees and officials

### Advanced Tracking
- **Player Tracking**: Real-time tracking of individual players across frames using ByteTrack
- **Ball Tracking**: Continuous tracking of ball movement throughout the match
- **Motion Trail Visualization**: Shows historical paths of ball movement

### Team Classification
- **Automatic Team Detection**: Uses SigLIP embeddings and KMeans clustering to classify players into teams
- **Visual Distinction**: Automatically assigns team colors (cyan and magenta by default)
- **Goalkeeper Assignment**: Intelligently assigns goalkeepers to their respective teams based on position proximity

### Pitch Analysis & Visualization
- **Pitch Keypoint Detection**: Detects field markings and boundaries
- **Perspective Transformation**: Projects camera view to overhead pitch view using homography
- **Voronoi Diagrams**: Visualizes ball possession and territorial control with smooth color transitions
- **Match Statistics**: Calculates player positions, ball location, and spatial analytics

## 📋 Prerequisites

### Required Credentials
Before running the notebook, you'll need to set up API keys:

1. **HuggingFace Token**
   - Visit [HuggingFace Settings](https://huggingface.co/settings)
   - Click `Access Tokens` → `New Token`
   - Store as `HF_TOKEN`

2. **Roboflow API Key**
   - Go to [Roboflow Settings](https://app.roboflow.com/settings/api)
   - Copy your API key
   - Store as `ROBOFLOW_API_KEY`

### Environment
- GPU support (NVIDIA CUDA 12.4+)
- Python 3.8+
- Jupyter Notebook or Google Colab

## 🚀 Installation

### Dependencies
```bash
pip install gdown inference-gpu
pip install git+https://github.com/roboflow/sports.git
pip install supervision>=0.23.0
pip install torch transformers scikit-learn umap-learn plotly
```

### GPU Configuration
Set ONNX Runtime to use CUDA:
```python
import os
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
```

## 📊 How It Works

### 1. Detection Models
- **Player Detection**: Uses pre-trained Roboflow model (football-players-detection-3zvbc/11)
- **Field Detection**: Uses Roboflow field detection model (football-field-detection-f07vi/14)

### 2. Team Classification Pipeline
```
Extract player crops → SigLIP embeddings → UMAP projection → KMeans clustering
```
- Collects player samples every 30 frames
- Generates 768-dimensional embeddings using SigLIP vision model
- Projects to 3D space with UMAP for visualization
- Performs 2-cluster KMeans to separate teams

### 3. Perspective Transformation
- Detects high-confidence pitch keypoints
- Calculates homography matrix between camera and pitch planes
- Transforms all detected objects to pitch coordinates
- Averages transformation matrices across frames for stability

### 4. Visualization Options
- **Standard View**: Bounding boxes with team colors and player IDs
- **Game-Style View**: Circles for players, triangles for ball
- **Pitch View**: Overhead perspective with detected players and ball
- **Voronoi Diagram**: Territory control visualization with team colors
- **Smooth Voronoi**: Blended transitions between team control areas

## 📁 Project Structure

```
player-tracking/
├── football_ai.ipynb          # Main Jupyter notebook with full pipeline
├── README.md                  # This file
└── .git/                       # Git repository
```

## 🔧 Usage

### Basic Usage
```python
# Import required libraries
import supervision as sv
from inference import get_model

# Load detection models
PLAYER_DETECTION_MODEL = get_model(model_id="football-players-detection-3zvbc/11", api_key=API_KEY)
FIELD_DETECTION_MODEL = get_model(model_id="football-field-detection-f07vi/14", api_key=API_KEY)

# Process video
frame_generator = sv.get_video_frames_generator("video.mp4")
for frame in frame_generator:
    # Run detection and tracking
    pass
```

### Complete Pipeline
The notebook includes a complete end-to-end pipeline that:
1. Loads video frames
2. Detects all players, ball, and officials
3. Tracks objects across frames
4. Classifies teams automatically
5. Projects objects to pitch coordinates
6. Creates various visualizations

## 📈 Key Metrics Generated

- **Player Positions**: Real-time coordinates of all players on the pitch
- **Ball Location**: Ball position in both camera and pitch coordinate systems
- **Territorial Control**: Voronoi diagrams showing which team controls which areas
- **Player Paths**: Historical movement trails of players and ball
- **Match Statistics**: Possession indicators and spatial distribution

## 🎨 Visualization Modes

### 1. Bounding Box View
Standard computer vision visualization with boxes around detected objects

### 2. Game-Style View
Circles for players (with team colors) and triangles for the ball

### 3. Pitch Overview
Top-down view of the field showing all detected players and ball positions

### 4. Voronoi Territory Control
Color-coded regions showing which team controls which areas of the pitch

### 5. Smooth Voronoi Blend
Enhanced Voronoi with smooth color transitions between team territories

## ⚙️ Configuration

### Soccer Pitch Configuration
The system uses `SoccerPitchConfiguration` which includes:
- Standard pitch dimensions (105m × 68m)
- Field markings and vertices
- Center circle and penalty areas
- Goal areas and corner positions

### Customizable Parameters
- `STRIDE`: Frame sampling interval (default: 30)
- `BALL_ID`: Model ID for ball (default: 0)
- `GOALKEEPER_ID`: Model ID for goalkeeper (default: 1)
- `PLAYER_ID`: Model ID for players (default: 2)
- `REFEREE_ID`: Model ID for referee (default: 3)
- `MAX_DISTANCE_THRESHOLD`: Ball tracking outlier threshold (default: 500px)

## 📊 Output Examples

- **Video Analysis**: Frame-by-frame detection and tracking visualization
- **Pitch Heatmaps**: Territory control visualization
- **Statistics**: Player distribution and ball movement analytics
- **Interactive Plots**: Clickable 3D team classification visualization

## 🔬 Technologies Used

- **Roboflow Inference**: Object detection and keypoint detection
- **Supervision**: Computer vision utilities and tracking
- **SigLIP**: Vision transformer embeddings for team classification
- **UMAP**: Dimensionality reduction for clustering visualization
- **KMeans**: Unsupervised clustering for team separation
- **ByteTrack**: Multi-object tracking algorithm
- **OpenCV**: Image processing and homography transformation
- **PyTorch**: Deep learning framework for embeddings

## 📝 Dataset

The notebook uses sample videos from the [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) Kaggle competition:
- Sample videos: 5 × ~20MB each
- Resolution: HD video format
- Content: Professional football/soccer matches

## 🐛 Troubleshooting

### GPU Not Available
Ensure GPU is enabled in Colab:
- Go to `Edit` → `Notebook settings`
- Set `Hardware accelerator` to `GPU`

### Missing Dependencies
```bash
pip install 'inference[transformers]' 'inference[sam]' 'inference[clip]' 'inference[gaze]' 'inference[grounding-dino]' 'inference[yolo-world]'
```

### Keypoint Detection Issues
Filter low-confidence keypoints using confidence threshold (default: 0.5)

## 📚 References

- [Roboflow Sports Repository](https://github.com/roboflow/sports)
- [Supervision Library](https://supervision.roboflow.com/)
- [SigLIP Model Documentation](https://huggingface.co/docs/transformers/en/model_doc/siglip)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)

## 🤝 Contributing

This is an educational project demonstrating state-of-the-art computer vision techniques in sports analytics. Feel free to adapt and extend it for your own use cases.

## 📄 License

This project builds upon the Roboflow sports repository and uses publicly available models. Please refer to individual libraries for their respective licenses.

## 🎓 Use Cases

- **Match Analysis**: Detailed analysis of player movement and team tactics
- **Performance Metrics**: Individual and team performance evaluation
- **Training Analytics**: Player movement patterns and spatial distribution
- **Broadcasting**: Enhanced commentary with automated statistics
- **Research**: Sports science and tactical analysis studies

---

**Note**: This notebook is optimized for Google Colab with GPU support and requires valid Roboflow and HuggingFace API keys.

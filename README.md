# 2D Autonomous Robot Simulator (`robot_nav`)

A complete, runnable 2D autonomous robot navigation simulator featuring OpenCV computer-vision map parsing, PyTorch CNN cell terrain classification, A* grid path planning with baselines & path smoothing, non-holonomic differential-drive kinematics, pure pursuit trajectory tracking, 2D LiDAR raycast perception with dynamic replanning, a 60 FPS Pygame GUI visualizer, and a multi-planner benchmark evaluation engine.

---

## Features & Highlights

- **Perception Pipeline (`robot_nav/perception/`)**:
  - Converts top-down RGB map images (hand-drawn or synthetic floorplans) into occupancy grids via OpenCV adaptive thresholding, morphological closing, contour detection, and 4-point perspective warping.
  - Automatically extracts Start (green circle) and Goal (red circle) grid coordinates using HSV color masking.
  - Small PyTorch CNN (`TerrainCNN`, <100k parameters) classifying 32x32 patches into `{Free: 1.0 cost, Obstacle: inf cost, Rough Terrain: 3.0 cost}` with pure CV intensity fallback.
  - Procedural synthetic patch generator (`dataset_gen.py`) and fast CPU training script (`train.py`, <3 min runtime).
- **Planning & Baselines (`robot_nav/planning/`)**:
  - A* search on weighted cost grids with 4/8-way connectivity, Manhattan/Euclidean/Octile heuristics, and tie-breaking epsilon.
  - Strict diagonal corner-cutting guard (diagonal step forbidden if adjacent cardinal cells are obstacles).
  - Path smoothing via Bresenham raycasting line-of-sight shortcutting.
  - Baseline planners: BFS, Dijkstra, and Greedy Best-First Search matching A* interface.
  - Dynamic Replanning (`replan.py`) with full event history logging.
- **Control & Kinematics (`robot_nav/control/`)**:
  - Continuous differential-drive kinematic model ($x, y, \theta, v, \omega$) with linear and angular velocity/acceleration limits.
  - Pure Pursuit steering controller with lookahead distance tracking and perpendicular cross-track error (CTE) calculation.
- **Simulation & Perception Rays (`robot_nav/sim/`)**:
  - 2D LiDAR / FOV raycaster finding obstacle intersections and dynamically revealing hidden map cells.
  - World simulation manager unifying ground-truth vs perceived maps, robot physics, and sensor scans.
  - 60 FPS Pygame GUI showing terrain heatmaps, A* expanded nodes heatmap, planned path, traveled trajectory, robot chassis + heading arrow, sensor ray fan, and live HUD overlay.
- **Evaluation & Benchmarks (`robot_nav/eval/`)**:
  - Trajectory metrics (path length, smooth length, distance traveled, mean/max CTE, expanded nodes, runtime, replan count).
  - 50-map benchmark runner evaluating 4 planners across 3 obstacle densities (10%, 20%, 30%), exporting `eval/results/benchmark_results.csv` and Matplotlib comparative plots (`path_cost.png`, `nodes_expanded.png`, `runtime.png`).
- **Configuration & Quality**:
  - Single centralized `SimConfig` dataclass (`robot_nav/config.py`) containing all system hyper-parameters.
  - Complete type hints on all public functions, Google-style docstrings, zero stub functions, and 100% passing `pytest` suite.

---

## Repo Structure

```
robot_nav/
├── config.py             # Central SimConfig dataclass for all tunable parameters
├── perception/
│   ├── map_from_image.py # OpenCV top-down map processing & HSV start/goal extraction
│   ├── cell_classifier.py# PyTorch CNN (<100k params) & pure CV fallback patch classifier
│   ├── dataset_gen.py    # Procedural synthetic 32x32 patch dataset generator
│   └── train.py          # Fast CPU training script (<3 min), confusion matrix, weight saver
├── planning/
│   ├── grid.py           # Occupancy and cost grid, 4/8-way connectivity, corner-cutting guards
│   ├── astar.py          # A* weighted search, Octile/Euclidean/Manhattan, path smoothing
│   ├── baselines.py      # BFS, Dijkstra, Greedy Best-First matching A* interface
│   └── replan.py         # Dynamic replanner and replan log manager
├── control/
│   ├── robot.py          # Kinematic differential drive model with velocity/acceleration limits
│   └── pure_pursuit.py   # Pure pursuit steering controller & cross-track error calculator
├── sim/
│   ├── sensor.py         # 2D LiDAR / FOV raycaster finding obstacle hits
│   ├── world.py          # Simulation world state (ground-truth vs perceived grid, robot, sensor)
│   ├── runner.py         # Interactive GUI and headless execution loops
│   └── render.py         # 60 FPS Pygame renderer with heatmaps, trail, sensor cone, HUD
├── eval/
│   ├── metrics.py        # Quantitative trajectory metrics and error calculation
│   └── benchmark.py      # 50-map benchmark runner exporting CSV & Matplotlib charts
├── maps/                 # Sample map generator and output map images
├── tests/                # Comprehensive Pytest test suite
├── main.py               # Central CLI entry point
├── requirements.txt      # Dependency specification
└── README.md             # Documentation
```

---

## Installation & Setup

1. **Clone & Setup Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quickstart & CLI Usage

### 1. Run Pipeline on Image Map
Converts a top-down image into an occupancy grid, plans path via A*, tracks path with Pure Pursuit, and displays Pygame window:
```bash
python main.py --image maps/sample1.png
```

### 2. Run on Procedurally Generated Random Map
Generates a 60x60 random grid map and runs the simulator:
```bash
python main.py --random --size 60x60
```

### 3. Run with Baseline Planners
Choose between `A*`, `BFS`, `Dijkstra`, or `Greedy`:
```bash
python main.py --random --planner Dijkstra
```

### 4. Headless Execution Mode (Fast non-GUI)
Run simulation without GUI display:
```bash
python main.py --image maps/sample1.png --headless
```

### 5. Train CNN Cell Classifier
Generates synthetic data and trains PyTorch `TerrainCNN` on CPU (<3 min):
```bash
python main.py --train
```

### 6. Run 50-Map Multi-Planner Benchmark
Evaluates BFS, Dijkstra, Greedy, and A* across 50 procedural maps at 10%, 20%, and 30% obstacle densities, saving CSV and Matplotlib plots to `eval/results/`:
```bash
python main.py --benchmark
```

---

## Interactive Pygame Controls

While the simulator window is active:
- `[SPACE]` : Pause / Resume simulation
- `[R]`     : Force dynamic path replan from robot's current pose
- `[N]`     : Generate a new random map
- `[ESC]`   : Quit simulation

---

## Running Unit Tests

Run the complete `pytest` test suite:
```bash
PYTHONPATH=. pytest
```
All 21 test cases cover:
- A* optimality against Dijkstra across 20 random grids
- Start == Goal and unreachable path conditions
- Diagonal corner-cutting prevention guards
- OpenCV image downsampling and start/goal detection
- PyTorch CNN architecture (<100k params) & CPU training
- Pure pursuit trajectory tracking convergence
- Lidar raycasting FOV bounds and dynamic obstacle replanning
- Metric computation and benchmark CSV export

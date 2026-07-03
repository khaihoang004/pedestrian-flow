# Pedestrian Flow

This project reproduces and analyzes the pedestrian flow experiments based on the hard-body and remote-action models, following the setup of Seyfried et al. (2006).


## Project Structure

```text
pedestrian-flow/
├─ analysis/
│  ├─ config.py          # all experiment settings and plotting configuration
│  ├─ helpers.py         # plotting helpers and utility functions
│  └─ plotting.py        # main script to run experiments and generate figures
├─ models/
│  ├─ empirical.py       # empirical reference data / empirical velocity curve
│  ├─ empirical_data.png # source image for empirical data extraction
│  ├─ social_force.py    # simulation models and core numerical routines
│  └─ plot_digitizer.py
├─ results/              # default generated figures folder path
├─ demo.html             # interactive browser demo 
├─ requirements.txt
└─ README.md
```


## Requirements

* Python 3.10 or newer
* Install dependencies:

```bash
pip install -r requirements.txt
```


## How to Run

### 1. Configure the experiment

Edit the experiment settings in:

```text
analysis/config.py
```

### 2. Run the simulation

From the project root directory, run:

```bash
python -m analysis.plotting
```

### 3. Check the results

By default, generated figures are saved in:

```text
results/
```

This folder already contains the figures generated from our experimental setup.
When you run new experiments, the newly generated figures will also be saved to the same folder.

The output directory can be changed in `analysis/config.py`.


## Extracting empirical data

The script `models/plot_digitizer.py` is used to digitize empirical data points from a plot image and save them to `models/empirical.py`.

### Run the digitizer

Using the default image:

```bash
python -m models.plot_digitizer
```

Using a custom image:

```bash
python -m models.plot_digitizer path/to/your_image.png
```

## Interactive HTML demo

The file `demo.html` provides a browser-based interactive demo of the pedestrian flow model.

Run the demo by open `demo.html` in a web browser.

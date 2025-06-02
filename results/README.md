# Ploting the Experiments Results

## Requirements

- Python 3.8+
- Gnuplot 5.4+ (can be installed by `sudo apt-get install gnuplot`)
- Eps2pdf (can be installed by `sudo apt-get install texlive-font-utils`)

## Usage

Each Fig subdirectory contains a `scripts` directory with the plotting script and a `processed` directory for the processed data.

To plot a figure, run the plotting script. For example, to plot Figure 7, run:

```bash
# Plot a figure
python results/fig7/scripts/plot.py
```


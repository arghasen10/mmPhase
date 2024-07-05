# <i>mmPhase</i> 

## Installation:

To install use the following commands.
```bash
git clone --recurse-submodules https://github.com/arghasen10/mmPhase.git
pip install -r requirements.txt
```

## Directory Structure

```
.
│   .gitignore
│   .gitmodules
│   arduino_receiver.c
│   command.py
│   configuration.py
│   create_merged_data.py
│   data_read_only_sensor.py
│   error_plot.py
│   estimate_velocity.py
│   generate_box.py
│   generate_doppler_velocity.py
│   generate_imu_baseline.py
│   generate_mat_file.py
│   generate_range_angle_plots.py
│   generate_results.py
│   helper.py
│   import_all.py
│   plot_box.py
│   README.md
│   socket_receiver.c
│   test_helper.py
│   vicon_estimate.py
│   visualize_heatmaps.py
│
└───milliEgo
```

## Description 

| Filename                     | Description                                                      | Expected Arguments                                               |
|------------------------------|------------------------------------------------------------------|------------------------------------------------------------------|
| arduino_receiver.c           | C source code for receiving data from Arduino.                    | Compilation and execution in a C environment.                    |
| command.py                   | Handles command-line operations or CLI interfaces.                | Command-line arguments or CLI options.                           |
| configuration.py             | Manages project configurations and settings.                      | Access to configuration parameters or constants.                |
| create_merged_data.py        | Merges multiple datasets into a single dataset.                  | Input datasets or files to merge.                               |
| data_read_only_sensor.py     | Reads data from a sensor in read-only mode.                      | Sensor data file or input configuration.                        |
| error_plot.py                | Generates error plots for data analysis.                         | Data for error analysis or comparison.                           |
| estimate_velocity.py         | Estimates velocity from sensor or simulation data.               | Sensor data or simulation inputs.                                |
| generate_box.py              | Generates box plots for visualizing data distributions.          | Data for plotting box plots.                                     |
| generate_doppler_velocity.py | Computes Doppler velocity from radar or ultrasound data.         | Raw radar or ultrasound data inputs.                             |
| generate_imu_baseline.py     | Generates IMU baselines or calibrations.                         | IMU sensor data or calibration parameters.                       |
| generate_mat_file.py         | Converts data or results into MATLAB (.mat) format.              | Data or results for conversion.                                  |
| generate_range_angle_plots.py| Generates plots of range and angle data.                         | Data for plotting range and angle data.                          |
| generate_results.py          | Generates results or reports from processed data.                | Processed data or analysis results.                              |
| helper.py                    | Contains utility functions or helpers for the project.           | Functions or utilities for general use across the project.       |
| import_all.py                | Imports necessary modules or components.                         | None                                                             |
| plot_box.py                  | Plots box plots for data visualization.                          | Data for plotting box plots.                                     |
| README.md                    | Project documentation providing setup instructions and overview. | None                                                             |
| socket_receiver.c            | C source code for receiving data over sockets.                   | Compilation and execution in a C environment.                    |
| test_helper.py               | Tests helper functions or modules.                               | Test cases and assertions.                                       |
| vicon_estimate.py            | Estimates parameters using Vicon motion capture data.            | Vicon motion capture data files or inputs.                       |
| visualize_heatmaps.py        | Visualizes heatmaps from sensor or simulation data.              | Data for generating and animating heatmaps.                      |

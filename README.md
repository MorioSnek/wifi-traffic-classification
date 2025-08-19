# Wi-Fi Traffic Classification
This repository contains code and resources for analyzing and classifying Wi-Fi traffic using classical and machine learning techniques. The project aims to identify different types of network traffic based on their characteristics.

## Setup notes
First of all, my environment is a MacBook Pro M1, running macOS Sequoia 15.6. I use Homebrew to manage my packages, and I have Python 3.13 installed.

To install the necessary packages and applications for the project:

```bash
brew install python
brew install wireshark
brew install wireshark-app
```
The difference between `wireshark` and `wireshark-app` is that the former installs the command-line tools, while the latter installs the GUI application. Additional tools that are part of Wireshark CLI are needed for this project.

### Virtual Environment
The scripts will run in a virtual environment. To set it up, you can use the following commands:

```bash
python -m venv .venv \
source .venv/bin/activate \
pip install -r requirements.txt
```

### Sniffing procedure
The `airport` command is deprecated, and the replacement command line tool `wdutil` does not serve the same purpose. Therefore, I had to use the GUI tool Wireless Diagnostics to enable the monitor mode in Wireshark.
```
Wireless Diagnostics -> Window -> Sniffer
```
After selecting the correct channel and bandwidth in the Wireless Diagnostics tool, it's possible to proceed with the capture in Wireshark.

### Folder structure
The project is structured as follows:

```.
wifi-traffic-classification/
├── data/
│   ├── plot_iat_avg_W15.png
│   ├── plot_iat_var_W15.png
│   ├── plot_pkt_size_avg_W15.png
│   ├── plot_pps_W15.png
│   ├── plot_rssi_W15.png
│   ├── plot_snr_W15.png
│   ├── plot_qos_composition_W15.png
│   ├── confusion_matrix_W15.png
│   └── plot_throughput_W15.png
│
├── model/
│   ├── features_labeled_W15.csv
│   ├── labels_by_time_W15.csv
│   └── features_W15.pcapng
│
├── src/
│   └── plots.py
│
├── private/
│   ├── traffic.csv
│   └── capture.pcapng
│
└── requirements.txt
```

The folders respect the following logic:
- `data/`: contains all the data files, mostly the generated plots.
- `private/`: contains sensitive data that will not be shared publicly. Contains the original capture and its processing results. Can be requested by the course instructor.
- `src/`: contains the source code for the project, including any scripts or modules used for analysis and visualization.
- `requirements.txt`: lists the Python packages required to run the project.

## Capture Processing
I personally found it easier to convert the whole Wireshark capture file into a format that can be processed by pandas easily, in particular `.csv`. To do so, it's possible to use the `tshark` command-line tool that comes with Wireshark. The following command can be used to filter and copy certain information from a `.pcap` or `.pcapng` file to a `.csv`:

```bash
tshark -r ./private/capture.pcapng \
  -Y "wlan.addr == XX:XX:XX:XX:XX:XX && wlan.fc.type=2" \
  -T fields -E header=y -E separator=, -E quote=d \
  -e frame.time_epoch \
  -e frame.len \
  -e wlan.sa -e wlan.da -e wlan.ta -e wlan.ra \
  -e wlan.fc.type -e wlan.fc.subtype \
  -e wlan.fc.pwrmgt \
  -e radiotap.dbm_antsignal \
  -e wlan.qos.priority \
  -e wlan_radio.snr \
> ./private/traffic.csv
```

### Classical analysis
The first part of this project focuses on classical traffic analysis without machine learning.
The script src/plots.py processes the CSV exported with tshark and computes several statistical features of Wi-Fi traffic for a given station MAC address.

Specifically, the script:
- Splits the traffic into fixed-size time windows (default: 15 seconds).
- Separates uplink and downlink packets.
- Computes per-window features:
    - RSSI [dBm]
    - SNR [dB]
    - Throughput [Mb/s]
    - Frames per second [FPS]
    - Average and variance of packet size
    - Average and variance of inter-arrival time
    - Quality of Service (QoS) distribution
- Extracts the Power Management bit to determine when the station is in sleep or awake mode, highlighting sleep periods in the plots.
- Produces time-series plots of all the above features, saved as .png images in the data/ folder.

This step provides a feature extraction and visualization pipeline that will serve as the foundation for the next stage, where machine learning techniques will be applied to classify user activities from encrypted Wi-Fi traffic.

### Machine Learning Approach
The machine learning side was developed using the scikit-learn library. In particular, its functioning is based on the following steps:
1.	**Feature extraction:** the relevant features are extracted from the raw traffic data, contained in model/features_W15.csv.

2.	**Data labeling:** the labels are assigned to the features based on the type of activity. This is done manually via the script `src/labeling.py`, which generates the files `model/labels_by_time_W15.csv` and `model/features_labeled_W15.csv`.

3.	**Model training and evaluation:** the dataset is split into training and test sets (based on the `TEST_SIZE`), a Random Forest classifier is trained on the training portion, and its performance is evaluated on the test set through accuracy, classification report, and confusion matrix.

The traffic classes considered in this work are streaming, web browsing, idle, and speedtest. These labels were chosen to cover the different user activities and to provide clear distinctions between different types of network behavior. The evaluation procedure therefore measures how effectively the model can separate these categories given only the statistical features extracted from the encrypted traffic.

## Results
For the first part, the plots generated provide initial insight regarding the RSSI and the SNR, just to be sure that the test is mostly valid. As plotted, the RSSI and the SNR are not only in the expected range, but also strictly correlated. This highlights the absence of noise during this experiment.
### Signal strenght

| RSSI                  | SNR                   |
| ----------------------------------- | ----------------------------------- |
| ![ps](/img/plot_rssi_W15.png) | ![fps](/img/plot_snr_W15.png)   |


### Throughput and Frames per second
The plots regarding the frames captured all highlight in red the areas where the station transceiver was off most of the time.

| Average frame size                  | Frames per second                   | Throughput                          |
| ----------------------------------- | ----------------------------------- | ----------------------------------- |
| ![ps](/img/plot_frame_size_avg_W15.png) | ![fps](/img/plot_fps_W15.png)   | ![thr](/img/plot_throughput_W15.png)  |

Starting the analysis from frames per second, frame size, and their obvious consequence, throughput:

- **Frames per second:** Strong peaks appear during active downlink traffic, especially when the station is awake. During sleep intervals (red shaded areas), the number of frames drops sharply, confirming reduced channel activity.

- **Average frame size:** Downlink frames are generally much larger than uplink ones, with consistent sizes above 1,000 bytes during high-traffic periods, while uplink traffic remains small and stable.

- **Throughput:** Traffic is heavily dominated by downlink, with bursts up to ~30 Mb/s in short intervals. During sleep phases, throughput collapses close to zero, showing the strong impact of power-saving states on data transfer.

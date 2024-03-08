# Transformer-based Spatio-Temporal Unsupervised Traffic Anomaly Detection in Aerial Videos

Authors: Tung Minh Tran, Doanh C. Bui, Tam V. Nguyen, Khang Nguyen

> Anomaly detection is an area of video analysis and plays an increasing role in ensuring safety, preventing risks, and guaranteeing quick response in intelligent surveillance systems. It has become popular research topics and piqued the interest of researchers in different communities, such as computer vision, machine learning, remote sensing, and data mining in recent years. This promotes novel mobile systems where drones are equipped with cameras to help people find better and more efficient solutions to automatically detect anomalies (e.g., car accidents, traffic congestion, street fighting) in traffic surveillance videos. However, anomaly detection methods are still rarely studied and developed in the remote sensing community due to anomalous events rarely occurring in real life, along with the high similarities between the objects of interest with small sizes, multi-scale objects, complex backgrounds of great variations, and high overlap between objects. Therefore, in order to fully exploit the spatio-temporal information for anomaly detection in traffic surveillance circumstances, we propose a future frame prediction network based on transformer architectures to detect abnormal events from drone videography in an unsupervised way. Our model treats consecutive video frames from an input clip and feeds features to a transformer encoder to capture spatial and temporal representations from the sequence, and then leverages a decoder to predict the next frame. Furthermore, an event is identified with high reconstruction error as an anomaly in the test phase. Thoroughly empirical studies demonstrate that our method achieves superior performance on the UIT-ADrone dataset and largely outperforms the state-of-the-art anomaly methods on the Drone-Anomaly dataset in aerial surveillance.

![image](https://hackmd.io/_uploads/rk2gKiuap.png)

## Progress

We aim to update following documents for our repo, will finish soon:

- [ ] Training document
- [ ] Inference document
- [ ] Checkpoints
- [ ] Dataset description

## Training

## Inference

## Citation

If this repository proves beneficial for your projects, we kindly request acknowledgment through proper citation:
```
@InProceedings{Tran_2024_TCSVT,
    author    = {Tran, Tung Minh and Bui, Doanh C. and Nguyen, Tam V. and Nguyen, Khang},
    title     = {Transformer-based Spatio-Temporal Unsupervised Traffic Anomaly Detection in Aerial Videos},
    journal = {IEEE Transactions on Circuits and Systems for Video Technology},
    month     = {},
    year      = {},
    pages     = {}
}
```
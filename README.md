# Transformer-based Spatio-Temporal Unsupervised Traffic Anomaly Detection in Aerial Videos

Authors: Tung Minh Tran, Doanh C. Bui, Tam V. Nguyen, Khang Nguyen

> Anomaly detection is an area of video analysis and plays an increasing role in ensuring safety, preventing risks, and guaranteeing quick response in intelligent surveillance systems. It has become popular research topics and piqued the interest of researchers in different communities, such as computer vision, machine learning, remote sensing, and data mining in recent years. This promotes novel mobile systems where drones are equipped with cameras to help people find better and more efficient solutions to automatically detect anomalies (e.g., car accidents, traffic congestion, street fighting) in traffic surveillance videos. However, anomaly detection methods are still rarely studied and developed in the remote sensing community due to anomalous events rarely occurring in real life, along with the high similarities between the objects of interest with small sizes, multi-scale objects, complex backgrounds of great variations, and high overlap between objects. Therefore, in order to fully exploit the spatio-temporal information for anomaly detection in traffic surveillance circumstances, we propose a future frame prediction network based on transformer architectures to detect abnormal events from drone videography in an unsupervised way. Our model treats consecutive video frames from an input clip and feeds features to a transformer encoder to capture spatial and temporal representations from the sequence, and then leverages a decoder to predict the next frame. Furthermore, an event is identified with high reconstruction error as an anomaly in the test phase. Thoroughly empirical studies demonstrate that our method achieves superior performance on the UIT-ADrone dataset and largely outperforms the state-of-the-art anomaly methods on the Drone-Anomaly dataset in aerial surveillance.

![overview](https://github.com/Tungufm/ASTT/assets/56221762/4040f597-d266-4e55-9505-8d47440141eb)

## Progress

We aim to update following documents for our repo, will finish soon:

- [ ] Training document
- [ ] Inference document
- [ ] Checkpoints
- [ ] Dataset description

## Training
> There are 59,186 frames for the training set and 147,005 frames for the test set in total 206,194 video frames. Notably, the training set only includes normal samples, whereas the test set consists of normal and abnormal patterns.

## Inference

## Dataset description
> UIT-ADrone dataset [1] consists of 3 different scenes captured by the complex traffic environment in Hochiminh city with 206,194 video frames (59,186 frames for the training set and 147,005 frames for the test set) in total, with size 1920 × 1080. Additionally, the dataset contains 592 training snippets and 905 testing snippets, totaling nearly 6.50 hours. The dataset has a total of 10 types of abnormal events, including crossing the road at the wrong lane, walking under the street, driving in the wrong roundabout, illegally driving on the sidewalk, illegal left turn/ turn right, illegally parking in the street, carrying bulky goods, parking on the sidewalk, driving in the opposite directions, and falling off motorcycles. In addition, the training snippets only have normal events, while testing snippets consist of both normal and unusual events. Additionally, the dataset is challenging for evaluation because of complex light conditions and camera movement. Furthermore, 63,485 ground truth annotations are provided for the testing set in the form of bounding boxes around each anomalous event in each extracted video frame, which helps evaluate the performance.

## References
[1] Tran, Tung Minh and Vu, Tu N and Nguyen, Tam V and Nguyen, Khang, “UIT-ADrone: A Novel Drone Dataset for Traffic Anomaly Detection”, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, IEEE, vol. 16, pp. 5590–5601, 2023.

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

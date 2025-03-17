# FLOCKD
## Federated Learning for Online Collaborative Knowledge and Decision-making
![image](https://github.com/user-attachments/assets/585d94dc-c6b3-4aef-b9b1-5a69341e9259)

Computing systems become continuously smaller and more pervasive. The Internet-of-Things (IoT) becomes more powerful with increasing computational power and storage on the devices. Utilising machine learning and specifically Deep Neural Networks (DNN), allows the devices to learn models of their environment and their current situations. Utilising these models allows for more accurate decision-making. However, training models for DNN is resource expensive and time consuming. Using Federated Learning (FL), a single DNN can be trained by multiple devices in a distributed fashion using only local information. FL combines models continuously without revealing the underlying information, guaranteeing the privacy of the used data.

Unfortunately, distributed IoT devices require individual and specialised models due to their location and inherent different perception of the world. Inspired by the Distributed Deep Neural Networks, we propose a novel approach in this project, separating DNNs into two parts: (i) a global/common, learned on all devices and improved using FL and (ii) an individual, specialised local part, specifically trained by each device. Using this separation, we can benefit from FL and train a considerable part of the DNN together with other devices while the local part remains specialised to the individual device.

Furthermore, operating in distributed IoT systems we can enable collaborative inference among the autonomous devices. For this, the output of the common part of the DNN can be forwarded to another device for the inference using the local specialised part of their respective DNN. The outcome of the local part can be returned to the original requester for affirmation.

While there are several application domains where sharing knowledge as well as support and feedback among different devices is useful (e.g. autonomous driving, multi-robot construction, or multi-drone exploration), the FLOCKD project will focus on object tracking and action recognition in visual sensor networks using smart cameras. Smart cameras have different perspectives on given situations while being able to operate autonomously, analyse and process imagery locally, and communicate and interact with other devices in their environment.

The project is executed in collaboration with the National University of Singapore, the University of Parma, Italy, and the Franklin and Marshall College, US

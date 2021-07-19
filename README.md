# Hide Data Footprints
This is an email spam/ham classifier that is trained using privacy-preserving federated learning.<br><br>
![concept](https://github.com/reenasheoran/Hide-Data-Footprints/blob/main/static/concept.png)
## Table of content
[Motivation](#Motivation)<br>
[Federated Learning](#Federated-Learning)<br>
[Installation](#Installation)<br>
[Data Collection](#Data-Collection)<br>
[Data Preparation](#Data-Preparation)<br>
[Project Lifecycle](#Project-Lifecycle)<br>
[Screen Shots](#Screen-Shots)<br>
[Conclusion](#Conclusion)<br>
[Application](#Application)<br>
[References](#References)<br>
## Motivation
The data owned by a single party may be very homogeneous, resulting in overfitting which negatively impacts accuracy when the model is applied to previously unseen data, i.e., poor generalizability. Utilizing data from diverse parties to train deep models can help mitigate this problem. However, collaborative model training may not be viable due to privacy concerns. Federated learning (FL), which incorporates privacy preservation techniques into collaborative model training, offers a potential solution to this challenge.
## Federated Learning
Federated learning is about a model going into a secure environment and learning how to solve a problem without needing the data to move anywhere.<br>
## Installation
This project is developed using python 3.8.8. If you are using any lower version of python then I recommend you to upgrade your python to the latest version by using pip command. Follow the steps below to run this project locally.<br>
**STEP 1:** git clone the repository
```
git clone https://github.com/reenasheoran/Hide-Data-Footprints.git
cd Hide-Data-Footprints
```
**STEP 2:** To execute the basic model training on centralized data run the following command.<br>
```
python Basic.py
```
**STEP 3:** To execute the federated model learning on decentralized data run the following command.<br>
```
python Federated.py
```
**STEP 4:** To check client data leakage by reverse engineering run the following command.<br>
```
python Client_data_leak.py
```
**STEP 5:** To execute the secure federated learning run the following command.<br>
```
python Encrypted_Federated.py
```
## Data Collection
The data was taken from http://www2.aueb.gr/users/ion/data/enron-spam/, which was in partially preprocessed form.The data originally contains 43908 emails with spam/ham labels.<br>
I created the vocabulary (size: 50635) and then made all the emails exactly of 500 words long by either trimming the email or padding it with <unk> tokens. Doing so made the final dataset square and then indexed the words to make the data ready for model training.<br>
## Data Preparation
I used the data in two ways and did splitting of data accordingly.<br>
1. For my basic model, I split the original data into training set (size: 41908) and test set (size: 2000) only.
2. For my federated models, I split the original data into 4 parts, Client_1 train data(size: 1000), client_2 training data(size: 1000) and Client_3 training_data (Size: 39908) and test data (size: 2000). I intentionally kept Client_3 training data big to have variations close to real-world scenario, as one company could have a large data at its first setup location and less data at its new setup locations.<br>
## Project Lifecycle
In this project, I began with the simple email spam/ham classifier that was created without the use of any standard library such as keras, tensorflow or Pytorch. Then, I took the steps as follows:- <br>
1. I created my first model using a centralized data to check the model accuracy level when the data is at one place (refer to Basic.py).<br>
2. In my second model I simulated a federated learning environment that has multiple data locations. The model is trained using the data available on those locations by sending and receiving model updates (refer to Federated.py).<br>
3. Then I tried reverse engineering on the model received from one of the client node, through federated learning, to get some idea about the respective client's training data (refer to Client_data_leak.py).<br>
4. Finally, I implemented a privacy-preserving federated learning in which model is first encrypted and then send to other node ( refer to Encrypted_Federated.py).<br>
## Screen Shots
## Conclusion
## Application
First, it means in order to participate in the 
deep learning supply chain, people don’t technically have to send their data to anyone. 
Valuable models in healthcare, personal management, and other sensitive areas can be 
trained without requiring anyone to disclose information about themselves. In theory, 
people could retain control over the only copy of their personal data (at least as far as deep 
learning is concerned).
This technique will also have a huge impact on the competitive landscape of deep learning 
in corporate competition and entrepreneurship. Large enterprises that previously wouldn’t 
(or couldn’t, for legal reasons) share data about their customers can potentially still earn 
revenue from that data.
## References

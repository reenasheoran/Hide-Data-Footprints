# Hide Data Footprints
This is an email spam/ham classifier that is trained using privacy-preserving federated learning.<br><br>
![concept](https://github.com/reenasheoran/Hide-Data-Footprints/blob/main/static/concept.png)
## Project Overview
In this project, I began with the simple email spam/ham classifier that was created without the use of any standard library such as keras, tensorflow or Pytorch. Then, I took the steps as follows:- <br>
1. I created my first model using a centralized data to check the model accuracy level when the data is at one place.<br>
2. In my second model I simulated a federated learning environment that has multiple data locations. The model is trained using the data available on those locations by sending and receiving model updates.<br>
3. Then I tried reverse engineering on the model received from one of the client node, through federated learning, to get some idea about the respective client's training data.<br>
4. Finally, I implemented a privacy-preserving federated learning in which model is first encrypted and then send to other node.<br>
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
**STEP 5:** To execute the secure federateed learning run the following command.<br>
```
python Encrypted_federated.py
```
## Data Collection
The data was taken from http://www2.aueb.gr/users/ion/data/enron-spam/, which was in partially processed form.<br>
I created the vocabulary (size: 50635) and then made all the emails exactly of 500 words long by either trimming the email or padding it with <unk> tokens. Doing so made the final dataset square.
then indexed the words to make the data ready for model training.<br>
## Data Preparation
I used the data in two ways and did splitting of data accordingly.<br>
1. For my basic model, I split the data into training and test set only, 

## Project Lifecycle
## Screen Shots
# Application
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

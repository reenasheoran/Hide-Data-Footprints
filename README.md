# Hide Data Footprints
This is an email spam/ham classifier that is trained using privacy-preserving federated learning.<br><br>
![concept](https://github.com/reenasheoran/Hide-Data-Footprints/blob/main/static/concept.png)
## Motivation
The data owned by a single party may be very homogeneous, resulting in overfitting which negatively impacts accuracy when the model is applied to previously unseen data, i.e., poor generalizability. Utilizing data from diverse parties to train deep models can help mitigate this problem. However, collaborative model training may not be viable due to privacy concerns. Federated learning (FL), which incorporates privacy preservation techniques into collaborative model training, offers a potential solution to this challenge.
## Federated Learning
Federated learning is about a model going into a secure environment and learning how to solve a problem without needing the data to move anywhere.<br>
## Installation
This project is developed using python 3.8.8. If you are using any lower version of python then I recommend you to upgrade your python to the latest version by using pip command. Follow the steps below to run this project locally.<br>
** STEP 1: ** git clone the repository
```
git clone https://github.com/reenasheoran/Hide-Data-Footprints.git
cd Hide-Data-Footprints
```
** STEP 2: ** To execute the basic model training on centralized data run the following command.<br>
```
python Basic.py
```
** STEP 3: ** To execute the federated model learning on decentralized data run the following command.<br>
```
python Federated.py
```
** STEP 4: ** To check client data leakage by reverse engineering run the following command.<br>
```
python Client_data_leak.py
```
** STEP 5: ** To execute the secure federateed learning run the following command.<br>
```
python Encrypted_federated.py
```
## Data Collection
The data has be taken from http://www2.aueb.gr/users/ion/data/enron-spam/, which is in preprocessed form.
## Data Preparation
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

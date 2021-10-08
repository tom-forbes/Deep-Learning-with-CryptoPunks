# Deep-Learning-with-CryptoPunks
Some Exploratory Deep Learning tasks on the multi-billion dollar CryptoPunk NFT collection.

The code to accompany my medium article on Depp Learning with CryptoPunks.

## Random DCGAN CryptoPunks
![dcgan copy](https://user-images.githubusercontent.com/39841498/133678008-bc114299-1d87-4fae-9fdd-6a41941b488c.gif)

## DCGAN Female & Male CryptoPunks
![dcgan_gender copy](https://user-images.githubusercontent.com/39841498/133677984-b3182cfb-09a4-412b-be65-1590dbf3fa43.gif)

## Setup

In command line
1. git clone https://github.com/tom-forbes/Deep-Learning-with-CryptoPunks : Clone the directory into your working environment
2. cd Deep-Learning-with-CryptoPunks/ : Make sure you are in the correct directory
3. python generate_data.py : run python script to call opensea API, saves CryptoPunk images and converts them to array, also saves relevant metadata. May take 10-20 minutes.
4. You should now have the data in your directory to play with the classification and DCGAN scripts, they will work better with a GPU so i would upload the scripts to  google colab if you havent got your own preference.


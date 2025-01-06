# LINUX COMMAND COMPLETION
This is a mini machine learning project developed using tensorflow, I used three model to compare performance (LSTM, GRU and RNN).
The project idea is to use one's command history to discover his command writing patter and assist him in future commands reducing the amount of time and effort needed to type linux commands. 
So if you're a lazy linux user who doesn't want to remember every single command, feel free to clone and use the project as you want (might need some adjustements though such as paths.. ).

## Installation:

### Requirements:
´´´
  pip install numpy pandas tensorflow keras_tuner scikit-learn matplotlib distutils-pytest 

´´´

### Deployment:

**1.** add the content of **config/bashrc_config** to your **.bashrc** file
**2.** create a  **.inputrc** file in your home
**3.** add the content of **config/inputrc_config** to your **.inputrc** file
**4.** source the **.bashrc** file
´´´ source .bashrc´´´
**5.** add the keybindings:
´´´ bind -f .inputrc´´´
**6. adjust** (modify paths, create envirements, all requirements, adjust keybindings ....) 
**7. be prepared to debug**

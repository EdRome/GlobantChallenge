# Globant Challenge

## Installation

`requirements.txt` file is included in the project. It uses 5 additional libraries to the installed in an python environment:

- nltk==3.8.1
- numpy==2.1.1
- pandas==2.2.3
- scikit_learn==1.5.2
- yellowbrick==1.5

The python environment should include the following libraries that exists by defect inside every python installation:

- re
- string
- subprocess
- itertools
- collections
- zipfile

## Usage

The project runs in console by calling the `main.py` file. It unzip the data inside `input_data` folder.

The project create a training set from the data the `input_data` of 2 thousand records. This is done to optimize the running time of the whole project. During data processing it seachs for wordnet.zip and stopwords.zip, in case they don't exists, then the project download them; be sure to have internet connection on the running machine to download them.

The output of the project is a list of topics with the top 5 words of each topic.

# Example

```
python main.py

[nltk_data] Downloading package wordnet to /content/working...
[nltk_data]   Package wordnet is already up-to-date!
Training score:  0.9296865413106109
Topics in text are:
 {0: ['time', 'water', 'across', 'team', 'interaction'], 1: ['chemical', 'theory', 'neutron', 'geometry', 'variety'], 2: ['performance', 'software', 'test', 'midir', 'light'], 3: ['chemical', 'water', 'proposed', 'interaction', 'polymer'], 4: ['property', 'may', 'aim', 'theory', 'one']}
```

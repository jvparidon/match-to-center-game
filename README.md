# match-to-center game
This repository contains data and code for the (multilingual) match-to-center embedding project.

Using participant data collected in a gamified triad match-to-center experiment we can construct a high-dimensional embedding space. The 
notebooks and code in this repository can create stimuli for the online experiment, process the collected data, and generate the embedding 
space.

The repository is structured as follows:
* `embedding_spaces.py`: Python module that contains code for:
    - the triad embedding algorithm (developed and implemented in TensorFlow by JP van Paridon)
    - the Kriegeskorte & Mur inverse MDS algorithm (reconstructed in Python by JP van Paridon)
    - code for analyzing the ability of the above two algorithms to recover latent structure in participant responses, including 
computing split-half correlations
    - code for cleaning and aggregating participant data from the online experiment
* `algorithm-analysis.ipynb`: notebook analyzing the triad embedding and inverse MDS algorithms' performance on three datasets (with 
varying numbers of concepts and participants)
* `50_concepts_set.csv`; `110_concepts_set.tsv`; `75_adjectives_animals_motionverbs.tsv`: files containing the three different stimulus 
sets used in the algorithm analyses; data for these stimuli is in the `data/` directory; data for the latter two sets were gathered using 
the match-to-center game, the first set was gathered by Aria in a separate study
* `create_feedbacksjs.ipynb`; `create_wordsimsjs.ipynb`: notebooks for generating the `.js` files containing stimuli and feedback trials 
for the match-to-center game; `.js` files can just be copied into the main directory for the game on the lab server
* `data/`: directory containing data from the match-to-center game
* `datasets/`: directory containing datasets that were used to generate the longlist of stimuli to be used in the match-to-center game; the 
final list still needs translation into various languages

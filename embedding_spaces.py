import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')  # disable GPU since this is slow on M1 MacBooks

from os import listdir, path
from itertools import chain
from random import sample
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from subs2vec.vecs import Vectors
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Lambda
from tensorflow.keras.regularizers import L1, L2
from tqdm.keras import TqdmCallback
from tqdm.auto import tqdm


# this requires having the fastText Common Crawl vectors, which can be downloaded from
# https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
# and subs2vec, which can be install using
# pip install subs2vec
vecs = Vectors('../embeddings/cc.en.300.vec', normalize=True, n=2e5)
vecs_dict = vecs.as_dict()

metric = 'cosine'


def preprocess_all(dichotomize=True):
    csvs = [fname for fname in listdir('data/50_concepts') if fname.endswith('.csv')]
    dfs = [pd.read_csv(path.join('data/50_concepts', csv)) for csv in csvs]
    for i in range(len(dfs)):
        dfs[i]['pp'] = csvs[i].replace('.csv', '')

    for i, df in enumerate(dfs):
        df.loc[df.slider_words == 'move this slider to the right',
               'catch_passed'] = df.loc[df.slider_words ==
                                        'move this slider to the right',
                                        'sliders'].astype(int) / 100
        df.loc[df.slider_words == 'move this slider to the left',
               'catch_passed'] = (100 - df.loc[df.slider_words ==
                                               'move this slider to the left',
                                               'sliders'].astype(int)) / 100
        pass_rate = df.catch_passed.sum() / 4
        df['catch_sum'] = pass_rate
        if pass_rate < .75:
            #print(f"participant {df['pp'].unique()} rejected with a pass rate of {pass_rate:.2f} (in {len(df)} trials).")
            pass
        else:
            dfs[i] = df
        
    # discard catch trials
    df = pd.concat(dfs)
    df = df.loc[df.question_type != 'catch']
    
    concepts_a = pd.read_csv('50_concepts_set.csv').sort_values('concept').reset_index(drop=True)
    concepts_a.concept = concepts_a.concept.apply(lambda x: x.split(' ')[-1])
    concepts_a.category = concepts_a.category.replace({'kitchenware': 'item', 'household item': 'item'})
    
    concepts_b = pd.read_csv('110_concepts_set.tsv', sep='\t')
    concepts_b.concept = concepts_b.concept.apply(lambda x: x.split(' ')[-1])
    
    concepts = concepts_a.merge(concepts_b, how='outer', on=['concept', 'category'])
    
    def compute_distances(trial):
        distances = []
        for k in trial['sliders'].split(','):
            coords = [0, 1 if int(k) > 50 else 0, 1]
            n = len(coords)
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist[i, j] = np.abs(coords[i] - coords[j])
            distances.append(dist)
        return distances
    
    def get_indices(trial, concepts):
        indices = []
        left = trial['left_words'].split(',')[0]
        right = trial['right_words'].split(',')[0]
        for slider in trial['slider_words'].split(','):
            words = [left, slider, right]
            n = len(words)
            idx = tuple([concepts.index[concepts.concept == word.split(' ')[-1]][0] for word in words])
            indices.append(idx)
        return indices


    def get_arrangs(df, concepts):
        indices = [get_indices(trial, concepts) for i, trial in df.iterrows()]
        arrangs = [compute_distances(trial) for i, trial in df.iterrows()]
        return (list(chain(*indices)), list(chain(*arrangs)))
    
    idx_a, arrangs_a = get_arrangs(df, concepts)
    
    
    df = pd.read_csv('data/triad_choices.tsv', sep='\t')
    
    def compute_distances(trial):
        if dichotomize:
            arr = np.array([
                [0, 1 - trial.chose_topword, 1],
                [1 - trial.chose_topword, 0, trial.chose_topword],
                [1, trial.chose_topword, 0]
            ])
        else:
            arr = np.array([
                [0, 100 - trial.pct_concept_b, 100],
                [100 - trial.pct_concept_b, 0, trial.pct_concept_b],
                [100, trial.pct_concept_b, 0]
            ])
        return [arr]
    
    def get_indices(trial, concepts):
        words = [trial.topword, trial.middleword, trial.bottomword]
        #print(words)
        return [tuple([concepts.index[concepts.concept == word.split(' ')[-1]][0] for word in words])]

    def get_arrangs(df, concepts):
        indices = [get_indices(trial, concepts) for i, trial in df.iterrows()]
        arrangs = [compute_distances(trial) for i, trial in df.iterrows()]
        return (list(chain(*indices)), list(chain(*arrangs)))
    
    idx_b, arrangs_b = get_arrangs(df, concepts)
    
    return np.vstack([idx_a, idx_b]), np.vstack([arrangs_a, arrangs_b]), concepts
    

# preprocess 50 concept ratings data
def preprocess_ratings():
    csvs = [fname for fname in listdir('data/50_concepts') if fname.endswith('.csv')]
    dfs = []
    
    for i, csv in enumerate(csvs):
        df = pd.read_csv(path.join('data/50_concepts', csv))
        df['pp'] = csvs[i].replace('.csv', '')
        df.loc[df.slider_words == 'move this slider to the right',
               'catch_passed'] = df.loc[df.slider_words ==
                                        'move this slider to the right',
                                        'sliders'].astype(int) / 100
        df.loc[df.slider_words == 'move this slider to the left',
               'catch_passed'] = (100 - df.loc[df.slider_words ==
                                               'move this slider to the left',
                                               'sliders'].astype(int)) / 100
        pass_rate = df.catch_passed.sum() / 4
        df['catch_sum'] = pass_rate
        df = df.loc[df.question_type != 'catch']
        #if pass_rate < .75:
        if pass_rate < 1.00:
            #print(f"participant {df['pp'].unique()} rejected with a pass rate of {pass_rate:.2f} (in {len(df)} trials).")
            pass
        else:
            dfs.append(df)
            
    print(f"usable participants: {len(dfs)}")
    print(f"usable trials: {sum(map(len, dfs)) * 4}")
    
    concepts = pd.read_csv('50_concepts_set.csv').sort_values('concept').reset_index(drop=True)
    
    def compute_distances(trial):
        distances = []
        for k in trial['sliders'].split(','):
            coords = [0, 1 if int(k) > 50 else 0, 1]
            n = len(coords)
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist[i, j] = np.abs(coords[i] - coords[j])
            distances.append(dist)
        return distances
    
    def get_indices(trial, concepts):
        indices = []
        left = trial['left_words'].split(',')[0]
        right = trial['right_words'].split(',')[0]
        for slider in trial['slider_words'].split(','):
            words = [left, slider, right]
            n = len(words)
            idx = tuple([concepts.index[concepts.concept == word][0] for word in words])
            indices.append(idx)
        return indices


    def get_arrangs(df, concepts):
        indices = [get_indices(trial, concepts) for i, trial in df.iterrows()]
        arrangs = [compute_distances(trial) for i, trial in df.iterrows()]
        return (list(chain(*indices)), list(chain(*arrangs)))
    
    idxarrangs = [get_arrangs(df, concepts) for df in dfs]
    
    return idxarrangs, concepts


def preprocess_triads(dichotomize=True):
    
    dfs = []
    for fname in sorted(listdir('data/110_concepts')):
        if fname.endswith('.csv'):
            df = pd.read_csv('data/110_concepts/' + fname)

            if len(df) > 150:

                # attach pp name
                df['pp'] = fname.replace('.csv', '')

                # check catch trials
                catch = df.loc[df.trial_type == 'catch'].copy()
                catch['catch_pass'] = catch.middleword == catch.chosenword
                passrate = np.mean(catch.catch_pass)
                df['catch_pass_rate'] = passrate
                #print(f"pp {fname.replace('.csv', '')} catch trial pass rate: {passrate:.2f}" +
                #      f" out of {len(catch)} catch and {len(df[df.trial_type == 'trial'])} overall trials")
                if passrate == 1.0:
                    df = df[df.trial_type == 'trial']
                    df = df[df.chosenword.notna()]
                    df['chose_topword'] = df.apply(
                        lambda row: 1 if (str(row.topword) == str(row.chosenword)) else 0, axis=1
                    )
                    dfs.append(df)
            
    print(f"usable participants: {len(dfs)}")
    print(f"usable trials: {sum(map(len, dfs))}")
    
    concepts = pd.read_csv('110_concepts_set.tsv', sep='\t')
    
    def compute_distances(trial):
        if dichotomize:
            arr = np.array([
                [0, 1 - trial.chose_topword, 1],
                [1 - trial.chose_topword, 0, trial.chose_topword],
                [1, trial.chose_topword, 0]
            ])
        else:
            arr = np.array([
                [0, 100 - trial.pct_concept_b, 100],
                [100 - trial.pct_concept_b, 0, trial.pct_concept_b],
                [100, trial.pct_concept_b, 0]
            ])
        return [arr]
    
    def get_indices(trial, concepts):
        words = [trial.topword, trial.middleword, trial.bottomword]
        return [tuple([concepts.index[concepts.concept == word][0] for word in words])]

    def get_arrangs(df, concepts):
        indices = [get_indices(trial, concepts) for i, trial in df.iterrows()]
        arrangs = [compute_distances(trial) for i, trial in df.iterrows()]
        return (list(chain(*indices)), list(chain(*arrangs)))
    
    idxarrangs = [get_arrangs(df, concepts) for df in dfs]
    concepts.concept = concepts.concept.apply(lambda x: x.split(' ')[-1])

    return idxarrangs, concepts


def process_feedback_trials():
    dfs = []
    
    for fname in sorted(listdir('data/75_concepts')):
        if fname.endswith('.csv'):
            df = pd.read_csv('data/75_concepts/' + fname)

            if len(df) > 150:

                # attach pp name
                df['pp'] = fname.replace('.csv', '')

                # check catch trials
                catch = df.loc[df.trial_type == 'catch'].copy()
                catch['catch_pass'] = catch.middleword == catch.chosenword
                passrate = np.mean(catch.catch_pass)
                df['catch_pass_rate'] = passrate
                #print(f"pp {fname.replace('.csv', '')} catch trial pass rate: {passrate:.2f}" +
                #      f" out of {len(catch)} catch and {len(df[df.trial_type == 'trial'])} overall trials")
                if passrate == 1.0:
                    df = df[df.trial_type == 'feedback']
                    dfs.append(df)
                
    for fname in sorted(listdir('data/110_concepts')):
        if fname.endswith('.csv'):
            df = pd.read_csv('data/110_concepts/' + fname)

            if len(df) > 150:

                # attach pp name
                df['pp'] = fname.replace('.csv', '')

                # check catch trials
                catch = df.loc[df.trial_type == 'catch'].copy()
                catch['catch_pass'] = catch.middleword == catch.chosenword
                passrate = np.mean(catch.catch_pass)
                df['catch_pass_rate'] = passrate
                #print(f"pp {fname.replace('.csv', '')} catch trial pass rate: {passrate:.2f}" +
                #      f" out of {len(catch)} catch and {len(df[df.trial_type == 'trial'])} overall trials")
                if passrate == 1.0:
                    df = df[df.trial_type == 'feedback']
                    dfs.append(df)
            
    print(f"usable participants: {len(dfs)}")
    print(f"usable trials: {sum(map(len, dfs))}")
    return pd.concat(dfs)


def process_regular_trials():
    dfs = []
    
    for fname in sorted(listdir('data/75_concepts')):
        if fname.endswith('.csv'):
            df = pd.read_csv('data/75_concepts/' + fname)

            if len(df) > 150:

                # attach pp name
                df['pp'] = fname.replace('.csv', '')

                # check catch trials
                catch = df.loc[df.trial_type == 'catch'].copy()
                catch['catch_pass'] = catch.middleword == catch.chosenword
                passrate = np.mean(catch.catch_pass)
                df['catch_pass_rate'] = passrate
                #print(f"pp {fname.replace('.csv', '')} catch trial pass rate: {passrate:.2f}" +
                #      f" out of {len(catch)} catch and {len(df[df.trial_type == 'trial'])} overall trials")
                if passrate == 1.0:
                    df = df[df.trial_type == 'trial']
                    df = df[df.chosenword.notna()]
                    df['chose_topword'] = df.apply(
                        lambda row: 1 if (str(row.topword) == str(row.chosenword)) else 0, axis=1
                    )
                    dfs.append(df)
                
    for fname in sorted(listdir('data/110_concepts')):
        if fname.endswith('.csv'):
            df = pd.read_csv('data/110_concepts/' + fname)

            if len(df) > 150:

                # attach pp name
                df['pp'] = fname.replace('.csv', '')

                # check catch trials
                catch = df.loc[df.trial_type == 'catch'].copy()
                catch['catch_pass'] = catch.middleword == catch.chosenword
                passrate = np.mean(catch.catch_pass)
                df['catch_pass_rate'] = passrate
                #print(f"pp {fname.replace('.csv', '')} catch trial pass rate: {passrate:.2f}" +
                #      f" out of {len(catch)} catch and {len(df[df.trial_type == 'trial'])} overall trials")
                if passrate == 1.0:
                    df = df[df.trial_type == 'trial']
                    df = df[df.chosenword.notna()]
                    df['chose_topword'] = df.apply(
                        lambda row: 1 if (str(row.topword) == str(row.chosenword)) else 0, axis=1
                    )
                    dfs.append(df)
            
    print(f"usable participants: {len(dfs)}")
    print(f"usable trials: {sum(map(len, dfs))}")
    return pd.concat(dfs)


def preprocess_novel_triads(dichotomize=True, verbose=False):
    
    dfs = []
    for fname in sorted(listdir('data/75_concepts')):
        df = pd.read_csv('data/75_concepts/' + fname)

        if len(df) > 150:

            # attach pp name
            df['pp'] = fname.replace('.csv', '')

            # check catch trials
            catch = df.loc[df.trial_type == 'catch'].copy()
            catch['catch_pass'] = catch.middleword == catch.chosenword
            passrate = np.mean(catch.catch_pass)
            df['catch_pass_rate'] = passrate
            if verbose:
                print(f"pp {fname.replace('.csv', '')} catch trial pass rate: {passrate:.2f}" +
                      f" out of {len(catch)} catch and {len(df[df.trial_type == 'trial'])} overall trials")
            if passrate >= 1.0:
                df = df[df.trial_type == 'trial']
                df = df[df.chosenword.notna()]
                df['chose_topword'] = df.apply(
                    lambda row: 1 if (str(row.topword) == str(row.chosenword)) else 0, axis=1
                )
                dfs.append(df)
            
    print(f"usable participants: {len(dfs)}")
    print(f"usable trials: {sum(map(len, dfs))}")
    
    concepts = (pd.read_csv('75_adjectives_animals_motionverbs.tsv', sep='\t')
                .rename(columns={'en_wordform': 'concept'}))
    
    def compute_distances(trial):
        if dichotomize:
            arr = np.array([
                [0, 1 - trial.chose_topword, 1],
                [1 - trial.chose_topword, 0, trial.chose_topword],
                [1, trial.chose_topword, 0]
            ])
        else:
            arr = np.array([
                [0, 100 - trial.pct_concept_b, 100],
                [100 - trial.pct_concept_b, 0, trial.pct_concept_b],
                [100, trial.pct_concept_b, 0]
            ])
        return [arr]
    
    def get_indices(trial, concepts):
        words = [trial.topword, trial.middleword, trial.bottomword]
        return [tuple([concepts.index[concepts.concept == word.split(' ')[-1]][0] for word in words])]

    def get_arrangs(df, concepts):
        indices = [get_indices(trial, concepts) for i, trial in df.iterrows()]
        arrangs = [compute_distances(trial) for i, trial in df.iterrows()]
        return (list(chain(*indices)), list(chain(*arrangs)))
    
    idxarrangs = [get_arrangs(df, concepts) for df in dfs]
    concepts.concept = concepts.concept.apply(lambda x: x.split(' ')[-1])

    return idxarrangs, concepts


# train center-match triad trial embeddings using a method analogous to Hebart et al. (2020)
def train_embeddings(idxarrangs, epochs=1, progress=True, loss='euclidean'):
    idx, arrangs = [list(chain.from_iterable(l)) for l in zip(*idxarrangs)]
    dim = int(np.max(idx) + 1)
    matrix = np.hstack([
        np.array(idx),
        np.array([1 if arrang[0][1] > .5 else 0 for arrang in arrangs]).reshape(-1, 1)
    ])

    @tf.function
    def vectorized_euclidean(y):
        norm = tf.nn.l2_normalize
        def score(x):
            # euclidean distance between normalized vectors
            return tf.sigmoid(tf.norm(norm(x[1]) - norm(x[0]), ord='euclidean') -
                              tf.norm(norm(x[1]) - norm(x[2]), ord='euclidean'))
        return tf.map_fn(score, y)
    
    @tf.function
    def vectorized_cosine(y):
        norm = tf.nn.l2_normalize
        def score(x):
            # dot product between normalized vectors (cosine distance)
            return tf.sigmoid(tf.tensordot(norm(x[1]), norm(x[2]), 1) - tf.tensordot(norm(x[1]), norm(x[0]), 1))
        return tf.map_fn(score, y)

    inputs = Input(shape=(3,))
    embeddings = Embedding(dim, dim, input_length=2,
                           embeddings_initializer='glorot_uniform',
                           #embeddings_regularizer=L2(1e-5)  # regularize?
                          )(inputs)
    if loss == 'euclidean':
        outputs = Lambda(vectorized_euclidean)(embeddings)
    elif loss == 'cosine':
        outputs = Lambda(vectorized_cosine)(embeddings)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    callbacks = [earlystop, TqdmCallback(verbose=0)] if progress else [earlystop]
    model.fit(x=matrix[:, :-1], y=matrix[:, -1], epochs=epochs, verbose=0, callbacks=callbacks, batch_size=128)
    dists = pairwise_distances(model.get_weights()[0], metric=metric)
    return dists


# print only 2 decimals
display2 = lambda x: display(x.round(2))

# generate uniform random number with 2 decimals
rand2 = lambda: np.around(np.random.uniform(), 2)

# compute root mean square of matrix
rms = lambda x: np.sqrt(np.mean(np.square(x)))

# scale matrix to a root mean square of 1
rms_scale = lambda x: x / rms(x)

# normalize vector
norm = lambda x: x / np.linalg.norm(x)

# compute cosine distance
cos = lambda a, b: np.dot(norm(a), norm(b))

# compute cosine projection
proj = lambda a, b, c: cos(c - a, b)

# grab upper triangle of square matrix
upper = lambda x: x[np.triu(np.ones(x.shape), k=1).astype(bool)]

# rank correlate distance matrices
def correlate_rdms(rdm1, rdm2):
    return spearmanr(upper(rdm1), upper(rdm2), nan_policy='omit')[0].round(2)


# compute inverse MDS using method 1 as described in Kriegeskorte & Mur (2012)
def inverse_mds(idxarrangs):
    indices, arrangs = [list(chain.from_iterable(l)) for l in zip(*idxarrangs)]
    n = np.max(indices) + 1
    
    # step 1: compute evidence weighted distance per pair    
    # step 2: make RDM from weighted average distances
    rdm = np.zeros((n, n))
    evidence = np.zeros((n, n))
    for i, arrang in enumerate(arrangs):
        idx = indices[i]
        for a, j in enumerate(idx):
            for b, k in enumerate(idx):
                if (a == 0) & (b == len(idx) - 1):
                    pass
                elif (b == 0) & (a == len(idx) - 1):
                    pass
                else:
                    #rdm[j, k] += arrang[a, b] * (arrang[a, b] ** 2)
                    #evidence[j, k] += arrang[a, b] ** 2
                    rdm[j, k] += arrang[a, b]
                    evidence[j, k] += 1
                
    mean_rdm = np.nan_to_num(rdm / evidence)
    
    def iterative_scaling(scaled_arrangs, rdm, itr, prior_rmsd=1e9):
        # step 3: scale RDM to RMS of 1
        rdm = rms_scale(rdm)

        # step 4: scale arrangement pair distances to have same RMS as arrangement pairs from RDM
        for i, arrang in enumerate(arrangs):
            idx = indices[i]
            rms_arrang = rms(arrang)
            rms_rdm = rms(rdm[idx, :][:, idx])
            scaled_arrangs[i] = arrang / (rms_arrang / rms_rdm)
        
        # step 5: create novel rdm
        prior_rdm = rdm
        rdm = np.zeros((n, n))
        for i, arrang in enumerate(arrangs):
            idx = indices[i]
            for a, j in enumerate(idx):
                for b, k in enumerate(idx):
                    if (a == 0) & (b == len(idx) - 1):
                        pass
                    elif (b == 0) & (a == len(idx) - 1):
                        pass
                    else:
                        #rdm[j, k] += arrang[a, b]
                        #rdm[j, k] += scaled_arrangs[i][a, b] * (arrang[a, b] ** 2)
                        rdm[j, k] += scaled_arrangs[i][a, b]
                    
        rdm = np.nan_to_num(rdm / evidence)
        
        # step 6: check if RMS of difference between new RDM and prior RDM is approaching 0
        # step 7: or return to step 3
        rmsd = rms(rdm - prior_rdm)
        itr += 1
        #print(f'iteration {itr}: {rmsd}')
        if prior_rmsd - rmsd > .001:
            return iterative_scaling(scaled_arrangs, rdm, itr, rmsd)
        else:
            return rdm, rmsd

    itr = 0
    scaled_arrangs = arrangs
    mds_rdm, rmsd = iterative_scaling(scaled_arrangs, mean_rdm, itr)
    return mds_rdm #, mean_rdm, rmsd


def evaluate_embeddings(dists, concepts, categories=False):
    decomp = MDS(2, dissimilarity='precomputed')
    #decomp = TSNE(2, metric='precomputed')
    if categories:
        concepts = concepts[concepts.category.isin(categories)]
        dists = dists[concepts.index, :][:, concepts.index]
        concepts = concepts.reset_index()
    concepts = concepts[['concept', 'category']]
    concepts[['x', 'y']] = decomp.fit_transform(dists)
    
    sns.set(style='white')
    fig = plt.figure(figsize=(6, 18))
    
    def label_points(df, xlabel, ylabel, value, ax):
        for i, point in df.iterrows():
            ax.text(point[xlabel]+.02, point[ylabel], str(point[value]), fontsize=6)

    ax = fig.add_subplot(311)
    
    # plot labeled points
    if 'category' in concepts.columns:
        sns.scatterplot(x='x', y='y', hue='category', data=concepts, ax=ax)
    else:
        sns.scatterplot(x='x', y='y', data=concepts, ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    label_points(concepts, 'x', 'y', 'concept', ax)
    
    ax = fig.add_subplot(312)
    
    # plot relationship with word embeddings
    concept_vecs = [vecs_dict[concept] for concept in concepts.concept]
    emb_rdm = pairwise_distances(concept_vecs, metric=metric)
    sns.regplot(x=upper(emb_rdm), y=upper(dists), scatter_kws={'alpha': 0.1}, ax=ax)
    #print(f'r(word embeddings, triad embeddings) = {correlate_rdms(emb_rdm, dists)}')
    ax.set(title=f"r(word embeddings, triad embeddings) = {correlate_rdms(emb_rdm, dists)}")

    ax = fig.add_subplot(313)
    
    # plot relationship with Hebart 2020
    labels = pd.read_csv('datasets/hebart/unique_id.txt', sep=' ', header=None, names=['concept'])
    hebart = pd.read_csv('datasets/hebart/spose_embedding_49d_sorted.txt', sep=' ', header=None)
    hebart['concept'] = labels.concept
    hebart = (
        hebart
        .merge(concepts, how='right', left_on='concept', right_on='concept')
        .drop(columns=['concept', 'category', 'x', 'y'])
        .dropna(axis=0, how='all')
        .dropna(axis=1)
    )
    hidx = hebart.index
    hebart_rdm = pairwise_distances(hebart, metric=metric)
    sns.regplot(x=upper(hebart_rdm), y=upper(dists[hidx, :][:, hidx]), scatter_kws={'alpha': 0.1}, ax=ax)
    #print(f'r(hebart norms, triad_embeddings) = {correlate_rdms(hebart_rdm, dists[hidx, :][:, hidx])}')
    ax.set(title=f"rank r(hebart norms, triad_embeddings) = {correlate_rdms(hebart_rdm, dists[hidx, :][:, hidx])}")
    
    plt.show()
    

def splithalf_corr(idxarrangs, splits=1, epochs=10, frac=1.0, loss='euclidean'):
    corrs = []
    for i in tqdm(range(splits)):
        idxarrangs_sample = sample(idxarrangs, int(len(idxarrangs) / (1 / frac)))
        a, b = train_test_split(idxarrangs_sample, test_size=len(idxarrangs_sample) // 2)
        adists = train_embeddings(a, epochs=epochs, progress=False, loss=loss)
        bdists = train_embeddings(b, epochs=epochs, progress=False, loss=loss)
        corrs.append(correlate_rdms(pairwise_distances(adists, metric=metric),
                                    pairwise_distances(bdists, metric=metric)))
    return np.mean(corrs).round(2), np.std(corrs).round(2)
    
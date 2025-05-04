# %% [markdown]
# ## Homework 3: Symbolic Music Generation Using Markov Chains

# %% [markdown]
# **Before starting the homework:**
# 
# Please run `pip install miditok` to install the [MiDiTok](https://github.com/Natooz/MidiTok) package, which simplifies MIDI file processing by making note and beat extraction more straightforward.
# 
# You’re also welcome to experiment with other MIDI processing libraries such as [mido](https://github.com/mido/mido), [pretty_midi](https://github.com/craffel/pretty-midi) and [miditoolkit](https://github.com/YatingMusic/miditoolkit). However, with these libraries, you’ll need to handle MIDI quantization yourself, for example, converting note-on/note-off events into beat positions and durations.

# %%
# run this command to install MiDiTok
#! pip install miditok

# %%
# import required packages
import random
from glob import glob
from collections import defaultdict

import numpy as np
from numpy.random import choice

from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile

# %%
# You can change the random seed but try to keep your results deterministic!
# If I need to make changes to the autograder it'll require rerunning your code,
# so it should ideally generate the same results each time.
random.seed(42)

# %% [markdown]
# ### Load music dataset
# We will use a subset of the [PDMX dataset](https://zenodo.org/records/14984509). 
# 
# Please find the link in the homework spec.
# 
# All pieces are monophonic music (i.e. one melody line) in 4/4 time signature.

# %%
midi_files = glob('PDMX_subset/*.mid')
len(midi_files)

# %% [markdown]
# ### Train a tokenizer with the REMI method in MidiTok

# %%
config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=midi_files)

# %% [markdown]
# ### Use the trained tokenizer to get tokens for each midi file
# In REMI representation, each note will be represented with four tokens: `Position, Pitch, Velocity, Duration`, e.g. `('Position_28', 'Pitch_74', 'Velocity_127', 'Duration_0.4.8')`; a `Bar_None` token indicates the beginning of a new bar.

# %%
# e.g.:
midi = Score(midi_files[0])
tokens = tokenizer(midi)[0].tokens
tokens[20:40]

# %% [markdown]
# 1. Write a function to extract note pitch events from a midi file; and another extract all note pitch events from the dataset and output a dictionary that maps note pitch events to the number of times they occur in the files. (e.g. {60: 120, 61: 58, …}).
# 
# `note_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of note pitch events (e.g. [60, 62, 61, ...])
# 
# `note_frequency()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to the number of times they occur, e.g {60: 120, 61: 58, …}

# %%
def note_extraction(midi_file):
    # Q1a: Your code goes here
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens
    note_pitches = []
    
    for t in tokens:
        if t.startswith("Pitch"):
            note_pitches.append(int(t.split('_')[1]))          
    
    # for track in midi.tracks:
    #     for note in track.notes:
    #         note_pitches.append(note.pitch)
    
    return note_pitches

# print(note_extraction(midi_files[0]))

# %%
from collections import Counter

def note_frequency(midi_files):
    # Q1b: Your code goes here
    freqs = Counter()   
    
    for file in midi_files:
        notes = note_extraction(file)
        freqs.update(Counter(notes))
        
    return dict(freqs)
        
# print(note_frequency(midi_files))

# %% [markdown]
# 2. Write a function to normalize the above dictionary to produce probability scores (e.g. {60: 0.13, 61: 0.065, …})
# 
# `note_unigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: a dictionary that maps note pitch events to probabilities, e.g. {60: 0.13, 61: 0.06, …}

# %%
def note_unigram_probability(midi_files):
    note_counts = note_frequency(midi_files)
    unigramProbabilities = {}
    
    # Q2: Your code goes here
    # ...
    total_count = sum(note_counts.values())
    unigramProbabilities = {pitch: count / total_count for pitch, count in note_counts.items()} 
       
    return unigramProbabilities

# print(note_unigram_probability(midi_files))

# %% [markdown]
# 3. Generate a table of pairwise probabilities containing p(next_note | previous_note) values for the dataset; write a function that randomly generates the next note based on the previous note based on this distribution.
# 
# `note_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramTransitions`: key: previous_note, value: a list of next_note, e.g. {60:[62, 64, ..], 62:[60, 64, ..], ...} (i.e., this is a list of every other note that occured after note 60, every note that occured after note 62, etc.)
# 
#   - `bigramTransitionProbabilities`: key:previous_note, value: a list of probabilities for next_note in the same order of `bigramTransitions`, e.g. {60:[0.3, 0.4, ..], 62:[0.2, 0.1, ..], ...} (i.e., you are converting the values above to probabilities)
# 
# `sample_next_note()`
# - **Input**: a note
# 
# - **Output**: next note sampled from pairwise probabilities

# %%
def note_bigram_probability(midi_files):
    bigramTransitions = defaultdict(list)
    bigramTransitionProbabilities = defaultdict(list)

    # Q3a: Your code goes here
    # ...
    bigramCounts = defaultdict(lambda: defaultdict(int))
    for midi_file in midi_files:
        pitches = note_extraction(midi_file)
        for i in range(len(pitches) - 1):
            bigramCounts[pitches[i]][pitches[i+1]] += 1
    
    for prev, next_dict in bigramCounts.items():
        next_nodes = list(next_dict.keys())
        next_node_counts = list(next_dict.values())
        next_node_probs = [count / sum(next_node_counts) for count in next_node_counts]
        
        bigramTransitions[prev] = next_nodes
        bigramTransitionProbabilities[prev] = next_node_probs
        

    return bigramTransitions, bigramTransitionProbabilities



# %%
def sample_next_note(note):
    # Q3b: Your code goes here
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    if note not in bigramTransitions:
        return None
    
    next_note = np.random.choice(bigramTransitions[note], p=bigramTransitionProbabilities[note])
    return next_note

# print(sample_next_note(64))

# %% [markdown]
# 4. Write a function to calculate the perplexity of your model on a midi file.
# 
#     The perplexity of a model is defined as 
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-1})))$
# 
#     where $p(w_1|w_0) = p(w_1)$, $p(w_i|w_{i-1}) (i>1)$ refers to the pairwise probability p(next_note | previous_note).
# 
# `note_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def note_bigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
     
    # Q4: Your code goes here
    # Can use regular numpy.log (i.e., natural logarithm)
    notes = note_extraction(midi_file)
    if len(notes) < 2:
        return None
    
    log_sum = 0
    
    for i in range(len(notes)):
        if i == 0:
            prob = unigramProbabilities.get(notes[i], 1e-6)
        else:
            if notes[i - 1] not in bigramTransitions:
                prob = 1e-6
            else:
                next_notes = bigramTransitions[notes[i - 1]]
                next_p = bigramTransitionProbabilities[notes[i - 1]]
                prob = next_p[next_notes.index(notes[i])] if notes[i] in next_notes else 1e-6
        log_sum += np.log(prob)
    
    perplexity = np.exp(-log_sum / len(notes))
    return perplexity

# print(note_bigram_perplexity(midi_files[0]))

# %% [markdown]
# 5. Implement a second-order Markov chain, i.e., one which estimates p(next_note | next_previous_note, previous_note); write a function to compute the perplexity of this new model on a midi file. 
# 
#     The perplexity of this model is defined as 
# 
#     $\quad \text{exp}(-\frac{1}{N} \sum_{i=1}^N \text{log}(p(w_i|w_{i-2}, w_{i-1})))$
# 
#     where $p(w_1|w_{-1}, w_0) = p(w_1)$, $p(w_2|w_0, w_1) = p(w_2|w_1)$, $p(w_i|w_{i-2}, w_{i-1}) (i>2)$ refers to the probability p(next_note | next_previous_note, previous_note).
# 
# 
# `note_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramTransitions`: key - (next_previous_note, previous_note), value - a list of next_note, e.g. {(60, 62):[64, 66, ..], (60, 64):[60, 64, ..], ...}
# 
#   - `trigramTransitionProbabilities`: key: (next_previous_note, previous_note), value: a list of probabilities for next_note in the same order of `trigramTransitions`, e.g. {(60, 62):[0.2, 0.2, ..], (60, 64):[0.4, 0.1, ..], ...}
# 
# `note_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def note_trigram_probability(midi_files):
    trigramTransitions = defaultdict(list)
    trigramTransitionProbabilities = defaultdict(list)
    
    # Q5a: Your code goes here
    # ...
    trigramCounts = defaultdict(lambda: defaultdict(int))
    
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        for i in range(len(notes) - 2):
            trigramCounts[(notes[i], notes[i + 1])][notes[i + 2]] += 1
    
    for prevs, next_dict in trigramCounts.items():
        next_notes = list(next_dict.keys())
        next_counts = list(next_dict.values())
        next_probs = [count / sum(next_counts) for count in next_counts]
        
        trigramTransitions[prevs] = next_notes
        trigramTransitionProbabilities[prevs] = next_probs

    return trigramTransitions, trigramTransitionProbabilities

# print(note_trigram_probability(midi_files))

# %%
def note_trigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    
    # Q5b: Your code goes here
    notes = note_extraction(midi_file)
    if len(notes) < 3:
        return None
    
    log_sum = 0
    
    for i in range(len(notes)):
        if i == 0:
            prob = unigramProbabilities.get(notes[i], 1e-6)
        elif i == 1:
            if notes[i] not in bigramTransitions or notes[0] not in bigramTransitions[notes[i]]:
                prob = 1e-6
            else:
                idx = bigramTransitions[notes[0]].index(notes[i])
                prob = bigramTransitionProbabilities[notes[0]][idx]
        else:
            if (notes[i - 2], notes[i - 1]) not in trigramTransitions or notes[i] not in trigramTransitions[(notes[i - 2], notes[i - 1])]:
                prob = 1e-6
            else:
                idx = trigramTransitions[(notes[i - 2], notes[i - 1])].index(notes[i])
                prob = trigramTransitionProbabilities[(notes[i - 2], notes[i - 1])][idx]
    
        log_sum += np.log(prob)
    
    perplexity = np.exp(-log_sum / len(notes))
    return perplexity

# print(note_trigram_perplexity(midi_files[0]))
    

# %% [markdown]
# 6. Our model currently doesn’t have any knowledge of beats. Write a function that extracts beat lengths and outputs a list of [(beat position; beat length)] values.
# 
#     Recall that each note will be encoded as `Position, Pitch, Velocity, Duration` using REMI. Please keep the `Position` value for beat position, and convert `Duration` to beat length using provided lookup table `duration2length` (see below).
# 
#     For example, for a note represented by four tokens `('Position_24', 'Pitch_72', 'Velocity_127', 'Duration_0.4.8')`, the extracted (beat position; beat length) value is `(24, 4)`.
# 
#     As a result, we will obtain a list like [(0,8),(8,16),(24,4),(28,4),(0,4)...], where the next beat position is the previous beat position + the beat length. As we divide each bar into 32 positions by default, when reaching the end of a bar (i.e. 28 + 4 = 32 in the case of (28, 4)), the beat position reset to 0.

# %%
duration2length = {
    '0.2.8': 2,  # sixteenth note, 0.25 beat in 4/4 time signature
    '0.4.8': 4,  # eighth note, 0.5 beat in 4/4 time signature
    '1.0.8': 8,  # quarter note, 1 beat in 4/4 time signature
    '2.0.8': 16, # half note, 2 beats in 4/4 time signature
    '4.0.4': 32, # whole note, 4 beats in 4/4 time signature
}

# %% [markdown]
# `beat_extraction()`
# - **Input**: a midi file
# 
# - **Output**: a list of (beat position; beat length) values

# %%
def beat_extraction(midi_file):
    # Q6: Your code goes here
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens
    beats = []
    pos = None
    
    for t in tokens:
        if t.startswith("Position"):
            pos = int(t.split("_")[1])
        if t.startswith("Duration"):
            length = duration2length[t.split("_")[1]]
            beats.append((pos, length))
    
    return beats

# print(beat_extraction(midi_files[0]))
    

# %% [markdown]
# 7. Implement a Markov chain that computes p(beat_length | previous_beat_length) based on the above function.
# 
# `beat_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatTransitions`: key: previous_beat_length, value: a list of beat_length, e.g. {4:[8, 2, ..], 8:[8, 4, ..], ...}
# 
#   - `bigramBeatTransitionProbabilities`: key - previous_beat_length, value - a list of probabilities for beat_length in the same order of `bigramBeatTransitions`, e.g. {4:[0.3, 0.2, ..], 8:[0.4, 0.4, ..], ...}

# %%
def beat_bigram_probability(midi_files):
    bigramBeatTransitions = defaultdict(list)
    bigramBeatTransitionProbabilities = defaultdict(list)
    
    # Q7: Your code goes here
    # ...
    bigramBeatCount = defaultdict(lambda: defaultdict(int))
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        for i in range(len(beats) - 1):
            bigramBeatCount[beats[i - 1][1]][beats[i][1]] += 1
    
    for prev, next_dict in bigramBeatCount.items():
        next_len = list(next_dict.keys())
        counts = list(next_dict.values())
        probs = [count / sum(counts) for count in counts]
        bigramBeatTransitions[prev] = next_len
        bigramBeatTransitionProbabilities[prev] = probs
    
    return bigramBeatTransitions, bigramBeatTransitionProbabilities

# print(beat_bigram_probability(midi_files))

# %% [markdown]
# 8. Implement a function to compute p(beat length | beat position), and compute the perplexity of your models from Q7 and Q8. For both models, we only consider the probabilities of predicting the sequence of **beat lengths**.
# 
# `beat_pos_bigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `bigramBeatPosTransitions`: key - beat_position, value - a list of beat_length
# 
#   - `bigramBeatPosTransitionProbabilities`: key - beat_position, value - a list of probabilities for beat_length in the same order of `bigramBeatPosTransitions`
# 
# `beat_bigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: two perplexity values correspond to the models in Q7 and Q8, respectively

# %%
def beat_pos_bigram_probability(midi_files):
    bigramBeatPosTransitions = defaultdict(list)
    bigramBeatPosTransitionProbabilities = defaultdict(list)

    
    # Q8a: Your code goes here
    # ...
    bigramBeatPosCount = defaultdict(lambda: defaultdict(int))
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        for pos, length in beats:
            bigramBeatPosCount[pos][length] += 1
    
    for pos, len_dict in bigramBeatPosCount.items():
        lens = list(len_dict.keys())
        counts = list(len_dict.values())
        probs = [count / sum(counts) for count in counts]
        
        bigramBeatPosTransitions[pos] = lens
        bigramBeatPosTransitionProbabilities[pos] = probs
        
    return bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities

# print(beat_pos_bigram_probability(midi_files))

# %%
def beat_unigram_probability(midi_files):
    freqs = Counter()
    
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        freqs.update(Counter([length for pos, length in beats]))
    
    total = sum(freqs.values())

    beatUnigramProbabilities = {length: count / total for length, count in freqs.items()} 
       
    return beatUnigramProbabilities

# print(beat_unigram_probability(midi_files))

# %%
def beat_bigram_perplexity(midi_file):
    bigramBeatTransitions, bigramBeatTransitionProbabilities = beat_bigram_probability(midi_files)
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    # Q8b: Your code goes here
    # Hint: one more probability function needs to be computed
    unigramBeatProbability = beat_unigram_probability(midi_files)
    
    beats = beat_extraction(midi_file)
    log_sum_Q7 = 0
    log_sum_Q8 = 0
    
    for i in range(len(beats)):
        if i == 0:
            prob_Q7 = unigramBeatProbability[beats[i][1]]           
        else:
            index_Q7 = bigramBeatTransitions[beats[i - 1][1]].index(beats[i][1])
            prob_Q7 = bigramBeatTransitionProbabilities[beats[i - 1][1]][index_Q7]
            
        index_Q8 = bigramBeatPosTransitions[beats[i][0]].index(beats[i][1])
        prob_Q8 = bigramBeatPosTransitionProbabilities[beats[i][0]][index_Q8]
        
        log_sum_Q7 += np.log(prob_Q7)
        log_sum_Q8 += np.log(prob_Q8)

    # perplexity for Q7
    perplexity_Q7 = np.exp(-log_sum_Q7 / len(beats))
    
    # perplexity for Q8
    perplexity_Q8 = np.exp(-log_sum_Q8 / len(beats))
    
    return perplexity_Q7, perplexity_Q8

# print(beat_bigram_perplexity(midi_files[0]))


# %% [markdown]
# 9. Implement a Markov chain that computes p(beat_length | previous_beat_length, beat_position), and report its perplexity. 
# 
# `beat_trigram_probability()`
# - **Input**: all midi files `midi_files`
# 
# - **Output**: two dictionaries:
# 
#   - `trigramBeatTransitions`: key: (previous_beat_length, beat_position), value: a list of beat_length
# 
#   - `trigramBeatTransitionProbabilities`: key: (previous_beat_length, beat_position), value: a list of probabilities for beat_length in the same order of `trigramBeatTransitions`
# 
# `beat_trigram_perplexity()`
# - **Input**: a midi file
# 
# - **Output**: perplexity value

# %%
def beat_trigram_probability(midi_files):
    trigramBeatTransitions = defaultdict(list)
    trigramBeatTransitionProbabilities = defaultdict(list)

    # Q9a: Your code goes here
    # ...
    trigramBeatCount = defaultdict(lambda: defaultdict(int))
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        for i in range(len(beats) - 1):
            trigramBeatCount[(beats[i - 1][1], beats[i][0])][beats[i][1]] += 1
    
    for prevs, len_dict in trigramBeatCount.items():
        lens = list(len_dict.keys())
        counts = list(len_dict.values())
        probs = [count / sum(counts) for count in counts]
        trigramBeatTransitions[prevs] = lens
        trigramBeatTransitionProbabilities[prevs] = probs
    
    return trigramBeatTransitions, trigramBeatTransitionProbabilities

# print(beat_trigram_probability(midi_files))


# %%
def beat_trigram_perplexity(midi_file):
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    trigramBeatTransitions, trigramBeatTransitionProbabilities = beat_trigram_probability(midi_files)
    # Q9b: Your code goes here
    
    log_sum = 0
    beats = beat_extraction(midi_file)
    for i in range(len(beats)):
        if i == 0:
            idx = bigramBeatPosTransitions[beats[i][0]].index(beats[i][1])
            prob = bigramBeatPosTransitionProbabilities[beats[i][0]][idx]     
        else:
            idx = trigramBeatTransitions[(beats[i - 1][1], beats[i][0])].index(beats[i][1])
            prob = trigramBeatTransitionProbabilities[(beats[i - 1][1], beats[i][0])][idx]
        
        log_sum += np.log(prob)
    
    perplexity = np.exp(-log_sum / len(beats))
    return perplexity

# print(beat_trigram_perplexity(midi_files[0]))

# %% [markdown]
# 10. Use the model from Q5 to generate N notes, and the model from Q8 to generate beat lengths for each note. Save the generated music as a midi file (see code from workbook1) as q10.mid. Remember to reset the beat position to 0 when reaching the end of a bar.
# 
# `music_generate`
# - **Input**: target length, e.g. 500
# 
# - **Output**: a midi file q10.mid
# 
# Note: the duration of one beat in MIDIUtil is 1, while in MidiTok is 8. Divide beat length by 8 if you use methods in MIDIUtil to save midi files.

# %%
def music_generate(length):
    # sample notes
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    
    # Q10: Your code goes here ...
    all_notes = list(unigramProbabilities.keys())
    all_notes_p = list(unigramProbabilities.values())
    prev_prev_note = np.random.choice(all_notes, p=all_notes_p)
    prev_note = np.random.choice(bigramTransitions[prev_prev_note], p=bigramTransitionProbabilities[prev_prev_note])
    sampled_notes = [prev_prev_note, prev_note]
    
    if length <= 2:
        sampled_notes = sampled_notes[:length]
    else:
        for _ in range(length - 2):
            key = (prev_prev_note, prev_note)
            if key in trigramTransitions:
                next_note = np.random.choice(trigramTransitions[key], p=trigramTransitionProbabilities[key])
            else:
                if prev_note in bigramTransitions:
                    next_note = np.random.choice(bigramTransitions[prev_note], p=bigramTransitionProbabilities[prev_note])
                else:    
                    next_note = np.random.choice(all_notes, p=all_notes_p)
            sampled_notes.append(next_note)
            prev_prev_note, prev_note = prev_note, next_note
    
    length2duration = {v: k for k, v in duration2length.items()}
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    current_position = 0
    bar_length = 32
    tokens = ["Bar_None"]
    
    for note in sampled_notes:
        if current_position >= bar_length:
            tokens.append("Bar_None")
            current_position -= bar_length
        
        if current_position not in bigramBeatPosTransitions:
            beat_length = random.choice(length2duration.keys())
        else:
            beat_length = np.random.choice(bigramBeatPosTransitions[current_position], p=bigramBeatPosTransitionProbabilities[current_position])
        duration = length2duration[beat_length]
        
        tokens.append(f"Position_{current_position}")
        tokens.append(f"Pitch_{note}")
        tokens.append("Velocity_127")
        tokens.append(f"Duration_{duration}")
        
        current_position += beat_length
    
    from miditok import TokSequence
    
    tokseq = TokSequence(tokens=tokens)
    score = tokenizer.decode([tokseq])
    score.dump_midi("q10.mid")   
        
    
    # sample beats
    sampled_beats = []
    
    # save the generated music as a midi file

# music_generate(200)




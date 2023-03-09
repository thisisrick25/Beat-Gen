import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import glob

# Load the dataset (in this case, we'll use the Classical Piano MIDI dataset)
data_dir = 'path/to/dataset'
files = glob.glob(data_dir + '/**/*.mid', recursive=True)

# Convert MIDI files to sequences of notes and durations
from music21 import converter, instrument, note, chord, stream

notes = []
durations = []

for file in files:
    midi = converter.parse(file)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
            durations.append(element.duration.quarterLength)
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
            durations.append(element.duration.quarterLength)

# Convert the notes and durations to integers
notes = sorted(list(set(notes)))
note_to_int = dict((note, number) for number, note in enumerate(notes))
int_to_note = dict((number, note) for number, note in enumerate(notes))

durations = sorted(list(set(durations)))
duration_to_int = dict((duration, number) for number, duration in enumerate(durations))
int_to_duration = dict((number, duration) for number, duration in enumerate(durations))

note_ints = [note_to_int[note] for note in notes]
duration_ints = [duration_to_int[duration] for duration in durations]

# Create the input and output sequences
sequence_length = 100 # Number of notes/durations in each input sequence
step = 1 # Step size between input sequences
note_sequences = []
duration_sequences = []
for i in range(0, len(notes) - sequence_length, step):
    note_seq_in = note_ints[i:i + sequence_length]
    duration_seq_in = duration_ints[i:i + sequence_length]
    note_seq_out = note_ints[i + sequence_length]
    duration_seq_out = duration_ints[i + sequence_length]
    note_sequences.append(note_seq_in)
    duration_sequences.append(duration_seq_in)

# Reshape the input sequences to the required format
num_samples = len(note_sequences)
input_shape = (sequence_length, 2) # 2 because each input sequence contains both a note and a duration
X = np.zeros((num_samples, *input_shape))
for i, (note_seq, duration_seq) in enumerate(zip(note_sequences, duration_sequences)):
    for j, (note_int, duration_int) in enumerate(zip(note_seq, duration_seq)):
        X[i, j, 0] = note_int
        X[i, j, 1] = duration_int

# Convert the output sequences to one-hot vectors
Y = keras.utils.to_categorical(note_sequences, num_classes=len(notes))

# Define the model
embedding_size = 128 # Size of the embedding layer
rnn_units = 256 # Number of units in the RNN layer
model = keras.Sequential([
    layers.Embedding(len(notes), embedding_size, input_length=sequence_length),
    layers.SimpleRNN(rnn_units),
    layers.Dense(len(notes), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
batch_size = 128
model.fit(X, Y, batch_size=batch_size, epochs=50)

# Generate new music
start_note = np.random.randint(0, len(notes) - sequence_length)
start_duration = np.random.randint(0, len(durations) - sequence_length)
start_sequence = np.zeros((1, sequence_length, 2))
start_sequence[:, :, 0] = note_ints[start_note:start_note + sequence_length]
start_sequence[:, :, 1] = duration_ints[start_duration:start_duration + sequence_length]

generated_notes = []
generated_durations = []
for i in range(500): # Generate 500 notes/durations
    predicted = model.predict(start_sequence)[0]
    predicted_note = np.argmax(predicted[0])
    predicted_duration = np.argmax(predicted[1])
    generated_notes.append(predicted_note)
    generated_durations.append(predicted_duration)
    start_sequence = np.roll(start_sequence, -1, axis=1)
    start_sequence[:, -1, 0] = predicted_note
    start_sequence[:, -1, 1] = predicted_duration

# Convert the generated notes and durations back to MIDI format and save the file
offset = 0
output_notes = []

for note_int, duration_int in zip(generated_notes, generated_durations):
    note_name = int_to_note[note_int]
    duration_name = int_to_duration[duration_int]
    if '.' in note_name:
        notes_in_chord = note_name.split('.')
        chord_notes = []
        for n in notes_in_chord:
            new_note = note.Note(int(n))
            new_note.storedInstrument = instrument.Piano()
            chord_notes.append(new_note)
        new_chord = chord.Chord(chord_notes)
        new_chord.quarterLength = duration_name
        output_notes.append(new_chord)
    else:
        new_note = note.Note(int(note_name))
        new_note.storedInstrument = instrument.Piano()
        new_note.quarterLength = duration_name
        output_notes.append(new_note)

    offset += duration_name

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='generated_music.mid')

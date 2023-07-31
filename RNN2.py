import numpy as np
import tensorflow as tf
from tensorflow import keras
from music21 import converter, instrument, note, chord, stream

def load_notes(file_path):
    """
    Load MIDI file and extract notes and durations as sequences of integers.
    """
    midi = converter.parse(file_path)
    notes = []
    durations = []
    for elem in midi.flat:
        if isinstance(elem, note.Note):
            notes.append(str(elem.pitch))
            durations.append(elem.duration.quarterLength)
        elif isinstance(elem, chord.Chord):
            notes.append('.'.join(str(n) for n in elem.normalOrder))
            durations.append(elem.duration.quarterLength)
    return notes, durations

def create_note_duration_dicts(notes, durations):
    """
    Create dictionaries to map between notes/durations and integer values.
    """
    unique_notes = sorted(set(notes))
    unique_durations = sorted(set(durations))
    note_to_int = dict((note, number) for number, note in enumerate(unique_notes))
    duration_to_int = dict((duration, number) for number, duration in enumerate(unique_durations))
    int_to_note = dict((number, note) for number, note in enumerate(unique_notes))
    int_to_duration = dict((number, duration) for number, duration in enumerate(unique_durations))
    return note_to_int, duration_to_int, int_to_note, int_to_duration

def create_sequences(notes, durations, sequence_length, note_to_int, duration_to_int):
    """
    Convert sequences of notes and durations to sequences of integer values.
    """
    note_ints = [note_to_int[note] for note in notes]
    duration_ints = [duration_to_int[duration] for duration in durations]
    input_sequences = []
    output_sequences = []
    for i in range(len(note_ints) - sequence_length):
        input_seq = np.zeros((sequence_length, 2))
        output_seq = np.zeros((2,))
        input_seq[:, 0] = note_ints[i:i+sequence_length]
        input_seq[:, 1] = duration_ints[i:i+sequence_length]
        output_seq[0] = note_ints[i+sequence_length]
        output_seq[1] = duration_ints[i+sequence_length]
        input_sequences.append(input_seq)
        output_sequences.append(output_seq)
    return np.array(input_sequences), np.array(output_sequences)

def create_model(sequence_length, num_notes, num_durations, embedding_size=128, rnn_units=256):
    """
    Create and compile the RNN model.
    """
    model = keras.models.Sequential([
        keras.layers.Embedding(num_notes, embedding_size, input_length=sequence_length),
        keras.layers.LSTM(rnn_units, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(rnn_units),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_notes, activation='softmax'),
        keras.layers.Dense(num_durations, activation='softmax')
    ])
    model.compile(loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'], optimizer='adam')
    return model

def train_model(model, X, Y, batch_size=128, epochs=50):
    """
    Train the RNN model on the input/output sequences.
    """
    model.fit(X, Y, batch_size=batch_size, epochs=epochs)

def generate_music(model, note_to_int, duration_to_int, int_to_note, int_to_duration, sequence_length, num_notes, num_durations):
    """
    Generate a sequence of notes and durations using the trained RNN model.
    """
    start_note = np.random.randint(0, num_notes - 1)
    start_duration = np.random.randint(0, num_durations - 1)
    generated_notes = [start_note]
    generated_durations = [start_duration]
    for i in range(500):
        input_seq = np.zeros((1, sequence_length, 2))
    input_seq[0, :, 0] = generated_notes[-sequence_length:]
    input_seq[0, :, 1] = generated_durations[-sequence_length:]
    prediction = model.predict(input_seq)[0]
    note_pred = np.argmax(prediction[0])
    duration_pred = np.argmax(prediction[1])
    generated_notes.append(note_pred)
    generated_durations.append(duration_pred)
    generated_notes = [int_to_note[note] for note in generated_notes]
    generated_durations = [int_to_duration[duration] for duration in generated_durations]
    return generate_midi(generated_notes, generated_durations)

    def sample(preds, temperature=1.0):
    # Helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_midi(notes, durations):
    """
    Convert sequences of notes and durations to a MIDI file.
    """
    output_notes = []
    offset = 0
    for i in range(len(notes)):
    pattern = notes[i]
    dur = durations[i]
    if '.' in pattern:
    notes_in_chord = pattern.split('.')
    chord_notes = []
    for current_note in notes_in_chord:
    new_note = note.Note(current_note)
    new_note.storedInstrument = instrument.Piano()
    chord_notes.append(new_note)
    new_chord = chord.Chord(chord_notes)
    new_chord.offset = offset
    new_chord.quarterLength = dur
    output_notes.append(new_chord)
    else:
    new_note = note.Note(pattern)
    new_note.offset = offset
    new_note.quarterLength = dur
    new_note.storedInstrument = instrument.Piano()
    output_notes.append(new_note)
    offset += dur
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='generated_music.mid')




# # Load the MIDI file and extract notes and durations
# notes, durations = load_notes('mozart.mid')

# # Create dictionaries to map between notes/durations and integer values
# note_to_int, duration_to_int, int_to_note, int_to_duration = create_note_duration_dicts(notes, durations)

# # Convert sequences of notes and durations to sequences of integer values
# X, Y = create_sequences(notes, durations, sequence_length, note_to_int, duration_to_int)

# # Create and compile the RNN model
# model = create_model(sequence_length, len(note_to_int), len(duration_to_int))

# # Train the RNN model on the input/output sequences
# train_model(model, X, Y)

# # Generate a sequence of notes and durations using the trained RNN model
# generated_music = generate_music(model, note_to_int, duration_to_int, int_to_note, int_to_duration, sequence_length, len(note_to_int), len(duration_to_int))
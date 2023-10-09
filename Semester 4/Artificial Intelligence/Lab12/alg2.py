import numpy as np
import tensorflow as tf
import os


class TextGenerator:
    def __init__(self):
        self.example = None
        self.target = None
        self.rnn_units = None
        self.embedding_dim = None
        self.vocab_size = None
        self.model = None
        self.text = ''
        self.examples_per_epoch = None
        self.char2idx = None
        self.idx2char = None
        self.vocab = None
        self.dataset = None
        self.checkpoint_dir = './training_checkpoints'
        self.BATCH_SIZE = 10
        self.BUFFER_SIZE = 10000
        self.EPOCHS = 30

    def getText(self):
        for i in range(1, 5):
            f = open("texts/" + str(i) + ".txt", "r")
            self.text = self.text + f.read() + ' '
            f.close()

    def adaptText(self):
        # The unique characters in the file
        self.vocab = sorted(set(self.text))
        # Creating a mapping from unique characters to indices
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)

        text_as_int = np.array([self.char2idx[c] for c in self.text])
        seq_length = 100
        self.examples_per_epoch = len(self.text) // (seq_length + 1)

        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

        dataset = sequences.map(self.split_input_target)

        self.dataset = dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)

    def split_input_target(self, chunk):
        inputText = chunk[:-1]
        targetText = chunk[1:]
        return inputText, targetText

    def build_model(self, vocab_size, embedding_dim, rnn_units, batch_size):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                      batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ])
        return model

    def build(self):
        # Length of the vocabulary in chars
        self.vocab_size = len(self.vocab)

        # The embedding dimension
        self.embedding_dim = 256

        # Number of RNN units
        self.rnn_units = 1024

        self.model = self.build_model(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            rnn_units=self.rnn_units,
            batch_size=self.BATCH_SIZE)
        for input_example_batch, target_example_batch in self.dataset.take(1):
            example_batch_predictions = self.model(input_example_batch)
            print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        self.target = target_example_batch
        self.example = example_batch_predictions

    def loss(self, labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    def train(self):
        example_batch_loss = self.loss(self.target, self.example)
        self.model.compile(optimizer='adam', loss=self.loss)

        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        history = self.model.fit(self.dataset, epochs=self.EPOCHS, callbacks=[checkpoint_callback])

    def generate_text(self, model, start_string, t):
        # Number of characters to generate
        num_generate = 100

        # Converting our start string to numbers (vectorizing)
        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperature results in more predictable text.
        # Higher temperature results in more surprising text.
        # Experiment to find the best setting.
        temperature = t

        # Here batch size == 1
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            # Pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(self.idx2char[predicted_id])

        return start_string + ''.join(text_generated)

    def getModel(self):
        model = self.build_model(self.vocab_size, self.embedding_dim, self.rnn_units, batch_size=1)
        model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
        model.build(tf.TensorShape([1, None]))
        return model

    def loadModel(self):
        return tf.keras.models.load_model('model.h5')


def run():
    generator = TextGenerator()

    generator.getText()
    generator.adaptText()
    generator.build()
    generator.train()
    model = generator.getModel()
    model.save("model.h5")
    # model = generator.loadModel()

    print(generator.generate_text(model, start_string="I am going", t=0.1))

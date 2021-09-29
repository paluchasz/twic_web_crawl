import json
import time

import numpy as np
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

from transformers import pipeline, DistilBertTokenizer, TFDistilBertModel, TFDistilBertForSequenceClassification,\
    AutoTokenizer, TFAutoModelForSequenceClassification, glue_convert_examples_to_features

DATA_DIR = Path('/Users/szymon.palucha/Desktop/sentiment_analysis/data/aclImdb')


def get_data(test=False, from_file=None):
    """Gets train data by default unless test=True. Loads from one file if path passed in."""
    if from_file:
        with open(from_file, 'r') as file:
            data = json.load(file)
        return data

    dir_path = DATA_DIR / 'test' if test else DATA_DIR / 'train'

    examples = []
    for label_dir in ['pos', 'neg']:
        for example in (dir_path / label_dir).glob('*txt'):
            ex_id, ex_rating = example.stem.split('_')
            with open(example, 'r') as file:
                text = file.read()
            examples.append({'text': text, 'id': int(ex_id), 'ex_rating': int(ex_rating),
                             'label': 1 if label_dir == 'pos' else 0})

    return examples


def predict(x, y, batch_size=100):
    """Using standard pipeline, disadvantage: not sure how many characters to limit to so < 512 tokens."""
    classifier = pipeline('sentiment-analysis')
    predictions = []
    batches_num = int(np.math.ceil(len(x) / batch_size))
    print('Number of batches {} of size {}'.format(batches_num, batch_size))
    start_time = time.time()
    for i in range(batches_num):
        print('Batch {}'.format(i))
        # Limit to 1000 characters
        predictions += classifier([d[:min(1000, len(d))] for d in x[i * batch_size:min((i + 1) * batch_size, len(x))]])

    print('Time taken {} minutes '.format((time.time() - start_time) / 60))
    predictions = [1 if p['label'] == 'POSITIVE' else 0 for p in predictions]
    accuracy = sum([1 for i in range(len(predictions)) if predictions[i] == y[i]]) / len(predictions)
    print('Accuracy: {}'.format(accuracy))
    return predictions


def predict_with_tokenizer(x, y, batch_size=100):
    """Need to predict in batches or will run out of memory"""
    predictions = []
    batches_num = int(np.math.ceil(len(x) / batch_size))
    print('Number of batches {} of size {}'.format(batches_num, batch_size))
    start_time = time.time()
    for i in range(batches_num):
        print('Batch {},'.format(i), end=' ')
        batch_start_time = time.time()
        input_tokens = tokenizer(x[i * batch_size:min((i + 1) * batch_size, len(x))], padding=True,
                                 truncation=True, max_length=512, return_tensors='tf')
        print('Tokenizer time: {},'.format(time.time() - batch_start_time), end=' ')
        tf_outputs = tf_model(input_tokens)
        predictions += tf.nn.softmax(tf_outputs.logits, axis=-1).numpy().tolist()
        print('Batch time: {}'.format(time.time() - batch_start_time))

    print('Time taken: {} minutes '.format((time.time() - start_time) / 60))
    print('Predictions: ', predictions)
    predictions = [1 if p[1] > 0.5 else 0 for p in predictions]
    print('Predictions: ', predictions)
    print('Actual:      ', y)
    accuracy = sum([1 for i in range(len(predictions)) if predictions[i] == y[i]]) / len(predictions)
    print('Accuracy: {}'.format(accuracy))
    return predictions


def tokenizer_example():
    """Simple example, define tokenizer and model in main"""
    inputs = tokenizer('This is a small positive example', padding=True, truncation=True, max_length=512,
                       return_tensors='tf')
    # Need to pad and truncate to 512 tokens, return_tensors to not give lists for input_ids/attention and mask and so
    # it can be passed to the model. Attention mask is 0 for pad tokens - to skip them from self attention
    # To pass in a pair of sentences (like question answering) pass in [[Sentence A, Sentence B], ...], or if just a
    # single pair pass in as two arguments, (a list is interpreted as a batch), so can also pass in two lists as two
    # arguments: one for all first sentences and one for all second sentences. Token_type_ids will indicate which
    # sentence each input sequence belongs too
    tf_outputs = tf_model(inputs)
    # First number in tuple is the score of negative, second is the score of positive sentiment, pass through softmax
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    # Or give label and will softmax loss already, can also output states
    tf_outputs_2 = tf_model(inputs, labels=tf.constant([0]), output_hidden_states=True, output_attentions=True)


def mrpc_example():
    """This is an example of Microsoft research paraphrase corpus, predicts whether two senteces are paraphrases"""
    original = 'Giraffes like Acacia leaves and hay, and they can consume 75 pounds of food a day.'
    paraphrase = 'A giraffe can eat up to 75 pounds of Acacia leaves and hay daily.'
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")
    tokens = tokenizer(original, paraphrase, return_tensors="tf")
    result = model(tokens)
    probabilities = tf.nn.softmax(result.logits)  # 1st: prob of not paraphrase, 2nd: prob of paraphrase
    return


def test_fine_tuned_model(model_dir):
    model = TFDistilBertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    tokens = tokenizer('This is a positive sentence', truncation=True, padding=True, return_tensors='tf')
    result = model(tokens)
    probabilities = tf.nn.softmax(result.logits)
    return


def tf_dataset_example():
    # Dataset example -------
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    data = tfds.load('glue/mrpc')
    # To view dataset object iterate through e.g. [x for x in data['train']]
    train_dataset_0 = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
    train_dataset_0 = train_dataset_0.shuffle(100).batch(32).repeat(2)
    # can view with [x for x in train_dataset]


def standard_training(x, y, epochs=1, batch_size=64, save_model_dir=None, stats_dir=None):
    start_time = time.time()
    # Make a validation set
    train_texts, val_texts, train_labels, val_labels = train_test_split(x, y, test_size=.2)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    print('Found all tokens in {} min'.format((time.time() - start_time) / 60))

    # Make tf dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
    train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))
    val_dataset = val_dataset.shuffle(1000).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss)
    print('Compiled model {}'.format((time.time() - start_time) / 60))

    # TensorBoard callback can save train losses after every batch. In terminal $ tensorboard --logdir=path_to_your_logs
    # tf_callback = tf.keras.callbacks.TensorBoard(log_dir=DATA_DIR / 'logs', update_freq='batch', write_images=True,
    #                                             histogram_freq=1, embeddings_freq=1)
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, batch_size=batch_size)  # callbacks=[tf_callback])
    print('Training took {}'.format(time.time() - start_time))
    if save_model_dir:
        print('Saving model')
        model.save_pretrained(save_model_dir)
    if stats_dir:
        json.dump(history.history, open(stats_dir, 'w'))


def custom_training(x, y, epochs=1, batch_size=64, val_batch_size=16, save_model_dir=None, stats_dir=None,
                    val_step_freq=None):
    """Implements a custom training loop so can validate more often"""
    start_time = time.time()
    train_texts, val_texts, train_labels, val_labels = train_test_split(x, y, test_size=0.05)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    print('Found all tokens in {} min'.format((time.time() - start_time) / 60))

    # Make tf dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels))
    train_dataset = train_dataset.shuffle(1000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels))
    val_dataset = val_dataset.shuffle(1000).batch(val_batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_losses = []
    val_losses = []
    steps = int(np.ceil(len(train_dataset)))
    if not val_step_freq:
        val_step_freq = int(steps / 10)
    for epoch in range(epochs):
        print('Starting epoch {}/{}'.format(epoch + 1, epochs))
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            step_time = time.time()
            train_loss = train_step(x_batch, y_batch, loss_func, optimizer)
            print('Training loss: {}'.format(train_loss))
            train_losses.append([epoch, step, steps, float(train_loss.numpy())])

            if step % val_step_freq == 0:
                val_time = time.time()

                # Validate every certain number of steps
                total_val_loss = validation_step(val_dataset, loss_func) / len(val_dataset)
                val_losses.append([epoch, step, steps, float(total_val_loss.numpy())])
                print('Validation loss: {}, time {}'.format(total_val_loss, time.time() - val_time))
            print('Step {}/{} Step time: {}'.format(step + 1, steps, time.time() - step_time))

    print('Training time: {}'.format(time.time() - start_time))
    if save_model_dir:
        print('Saving model')
        model.save_pretrained(save_model_dir)
    if stats_dir:
        print('Saving stats')
        json.dump({'train_losses': train_losses, 'val_losses': val_losses}, open(stats_dir, 'w'))


@tf.function
def validation_step(val_dataset, loss_func):
    total_loss = tf.constant(0, dtype=tf.float32)
    for i, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        val_logits = model(x_batch_val, training=False)
        val_loss = loss_func(y_batch_val, val_logits.logits)
        total_loss += val_loss

    return total_loss # tf.math.divide(total_loss, tf.constant(len(val_dataset), dtype=tf.float32))


@tf.function  # comment out to run in debug
def train_step(x, y, loss_func, optimizer):
    """Implements one training step. Use decorator for fast performance using graphs. Seems to halve the train time!"""
    # Open a GradientTape to records operations during the forward pass which enables auto-differentiation
    with tf.GradientTape() as tape:
        # Run forward pass, calculate loss
        train_logits = model(x, training=True)
        train_loss = loss_func(y, train_logits.logits)
    # Use the tape tp automatically retrieve the gradients of the trainable variables with respect to the loss
    grads = tape.gradient(train_loss, model.trainable_weights)
    # Run one step of gradient descent by updating the values of the variables to minimise the loss
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return train_loss


if __name__ == '__main__':
    training_data = get_data(from_file=DATA_DIR / 'training_data.json')
    test_data = get_data(from_file=DATA_DIR / 'test_data.json')

    np.random.seed(1)
    np.random.shuffle(training_data)
    x_train_, y_train_ = [list(x) for x in zip(*[[t['text'], t['label']] for t in training_data])]

    x_test_, y_test_ = [list(x) for x in zip(*[[t['text'], t['label']] for t in test_data])]

    # Using existing pipeline classifier ---------------
    # predict(x_test, y_test)

    # Using tokenizer - equivalent to above  -----------
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    # tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # # For this model the Auto Model/Tokenizer is equivalent to:
    # tf_model = TFDistilBertForSequenceClassification.from_pretrained(model_name)
    # # A linear layer on top of the pooled output (of DistilBert)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    tokenizer_example()
    # predict_with_tokenizer(x_test, y_test)

    test = False
    if test:
        # test_fine_tuned_model(DATA_DIR / 'my_sentiment_model/')
        tf_model = TFDistilBertForSequenceClassification.from_pretrained(DATA_DIR / 'my_sentiment_model_e1b64')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        predict_with_tokenizer(x_test_[:10], y_test_[:10])

    fine_tune_model = True
    if fine_tune_model:
        # Fine tuning ------------------------------------
        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # custom_training(x_train_, y_train_, save_model_dir=DATA_DIR / 'my_sentiment_model_valid_2',
        #                 stats_dir=DATA_DIR / 'my_sentiment_model_valid_2_losses.json', batch_size=16)
        custom_training(x_train_[:10], y_train_[:10], batch_size=2, val_step_freq=2)

        # standard_training(x_train_[:50], y_train_[:50])

    a = 1


# Standard sentiment-analysis pipeline: accuracy = 0.867 (1000 char limit)
# Using tokenizer : Time taken: 142.12985572020213 minutes, Accuracy: 0.88871485943775

# Fined_tuned, epochs: 3, batch_size=16, steps_per_epoch=50; Time: 153min, Accuracy: 0.861285
# Fine_tuned, epochs: 1, batch_size=64, steps_per_epoch=None (ie 25k/64=391), Time >2h Accuracy: 0.91376

#   1/157 [..............................] - ETA: 1:08:25 - loss: 0.6964
#   2/1563 [..............................] - ETA: 6:41:59 - loss: 0.6910
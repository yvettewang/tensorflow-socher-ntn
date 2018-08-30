import tensorflow as tf
import ntn_input
import ntn
import params
import numpy as np
import numpy.matlib
import random
import datetime


def data_to_indexed(data, entities, relations):
    entity_to_index = {entities[i]: i for i in range(len(entities))}
    relation_to_index = {relations[i]: i for i in range(len(relations))}
    indexed_data = [(entity_to_index[data[i][0]], relation_to_index[data[i][1]], \
                     entity_to_index[data[i][2]]) for i in range(len(data))]
    return indexed_data

def get_train_batch(batch_size, data, num_entities, corrupt_size):
    random_indices = random.sample(range(len(data)), batch_size)
    # data[i][0] = e1, data[i][1] = r, data[i][2] = e2, random=e3 (corrupted)
    batch = [(data[i][0], data[i][1], data[i][2], random.randint(0, num_entities - 1)) \
             for i in random_indices for j in range(corrupt_size)]
    return batch

def get_test_batch(batch_size, data):
    random_indices = random.sample(range(len(data)), batch_size)
    # data[i][0] = e1, data[i][1] = r, data[i][2] = e2
    batch = [(data[i][0], data[i][1], data[i][2]) for i in random_indices]
    return batch

def split_train_batch(data_batch, num_relations):
    batches = [[] for i in range(num_relations)]
    for e1, r, e2, e3 in data_batch:
        batches[r].append((e1, e2, e3))
    return batches

def split_test_batch(data_batch, num_relations):
    batches = [[] for i in range(num_relations)]
    for e1, r, e2 in data_batch:
        batches[r].append((e1, e2))
    return batches

def fill_feed_dict(batches, train_both, batch_placeholders, label_placeholders, corrupt_placeholder):
    feed_dict = {corrupt_placeholder: [train_both and np.random.random() > 0.5]}
    for i in range(len(batch_placeholders)):
        feed_dict[batch_placeholders[i]] = batches[i]
        feed_dict[label_placeholders[i]] = [[0.0] for j in range(len(batches[i]))]
    return feed_dict


def run_training():
    print("Begin!")
    # python list of (e1, R, e2) for entire training set in string form
    print("Load training data...")
    # shape of raw training data: (112581, 3)
    raw_training_data = ntn_input.load_training_data(params.data_path)
    raw_dev_data = ntn_input.load_dev_data(params.data_path)
    raw_test_data = ntn_input.load_test_data(params.data_path)

    print("Load entities and relations...")
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_relations(params.data_path)
    num_entities = len(entities_list)  # entity: 38696
    num_relations = len(relations_list)  # relations: 11
    # python list of (e1, R, e2) for entire training set in index form
    indexed_training_data = data_to_indexed(raw_training_data, entities_list, relations_list)
    indexed_dev_data = data_to_indexed(raw_dev_data, entities_list, relations_list)
    indexed_test_data = data_to_indexed(raw_test_data, entities_list, relations_list)

    print("Load embeddings...")
    # shape of word embeds: 67447, 100; number of entities: 38696
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)

    num_iters = params.num_iter
    batch_size = params.batch_size
    corrupt_size = params.corrupt_size
    slice_size = params.slice_size

    with tf.Graph().as_default():
        print("Starting to build graph " + str(datetime.datetime.now()))
        batch_placeholders = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_' + str(i)) for i in
                              range(num_relations)]
        label_placeholders = [tf.placeholder(tf.float32, shape=(None, 1), name='label_' + str(i)) for i in
                              range(num_relations)]
        corrupt_placeholder = tf.placeholder(tf.bool, shape=1)
        train_inference = ntn.inference(batch_placeholders, corrupt_placeholder, init_word_embeds,
                                  entity_to_wordvec, num_entities, num_relations, slice_size, batch_size,
                                  False, label_placeholders)
        test_inference = ntn.inference(batch_placeholders, corrupt_placeholder, init_word_embeds,
                                  entity_to_wordvec, num_entities, num_relations, slice_size, batch_size,
                                  True, label_placeholders)
        train_loss = ntn.loss(train_inference, params.regularization)
        training = ntn.training(train_loss, params.learning_rate)


        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(tf.trainable_variables())

        # training
        for i in range(1, num_iters):
            print("Starting iter " + str(i) + " " + str(datetime.datetime.now()))
            data_train_batch = get_train_batch(batch_size, indexed_training_data, num_entities, corrupt_size)
            relation_train_batches = split_train_batch(data_train_batch, num_relations)
            data_dev_batch = get_test_batch(batch_size, indexed_dev_data)
            relation_dev_batches = split_test_batch(data_dev_batch, num_relations)

            feed_dict_training = fill_feed_dict(relation_train_batches, params.train_both, batch_placeholders,
                                       label_placeholders, corrupt_placeholder)
            feed_dict_dev = fill_feed_dict(relation_dev_batches, params.train_both, batch_placeholders,
                                               label_placeholders, corrupt_placeholder)

            _, train_loss_value, train_eval_value = sess.run([training, train_loss, ntn.eval(train_inference)], feed_dict=feed_dict_training)
            dev_eval_value = sess.run(ntn.eval(test_inference), feed_dict=feed_dict_dev)

            print("Iter {}, Training data loss = {}".format(i, train_eval_value))
            print("Iter {}, Dev data loss = {}".format(i, dev_eval_value))

            if i % params.save_per_iter == 0:
                saver.save(sess, params.output_path + "/" + params.data_name + str(i) + '.sess')
                print("Model saved at iter {}".format(i))

        # test
        data_test_batch = get_test_batch(batch_size, indexed_test_data)
        relation_test_batches = split_test_batch(data_test_batch, num_relations)
        feed_dict_testing = fill_feed_dict(relation_test_batches, params.train_both, batch_placeholders,
                                               label_placeholders, corrupt_placeholder)
        test_eval_value = sess.run(ntn.eval(test_inference), feed_dict=feed_dict_testing)
        print("Final Test Accuracy = {}".format(test_eval_value))

def main(argv):
    run_training()


if __name__ == "__main__":
    tf.app.run()

import tensorflow as tf
import params
import ntn_input
import random


# returns a (batch_size * corrupt_size, 2) vector corresponding to [g(T^i), g(T_c^i)] for all i
def inference(batch_placeholders, corrupt_placeholder, init_word_embeds, entity_to_wordvec, \
              num_entities, num_relations, slice_size, batch_size, is_eval, label_placeholders):
    print("Beginning building inference:")
    print("Creating variables")
    d = 100  # embed_size
    k = slice_size
    ten_k = tf.constant([k])
    num_words = len(init_word_embeds)

    E = tf.Variable(init_word_embeds)  # d=embed size
    print("tf E shape: {}".format(E.get_shape()))
    W = [tf.Variable(tf.truncated_normal([d, d, k])) for r in range(num_relations)]
    V = [tf.Variable(tf.truncated_normal([k, 2 * d])) for r in range(num_relations)]
    b = [tf.Variable(tf.truncated_normal([k, 1])) for r in range(num_relations)]
    U = [tf.Variable(tf.ones([1, k])) for r in range(num_relations)]

    print("Calcing ent2word")
    # python list of tf vectors: i -> list of word indices cooresponding to entity i
    ent2word = [tf.cast(tf.constant(entity_i) - 1, tf.int32) for entity_i in entity_to_wordvec]  # 38696

    print("Calcing entEmbed...")
    # take average embeddings as the entity embedding
    emtEmbed = tf.stack([tf.reduce_mean(tf.gather(E, entword), 0) for entword in ent2word])
    # tf.gather(): (3, 100), tf.reduce_mean(): (100,)
    print(emtEmbed.get_shape())  # (38696, 100)

    predictions = list()
    print("Beginning relations loop")
    for r in range(num_relations):
        print("Relations loop " + str(r))
        e1, e2, e3 = tf.split(tf.cast(batch_placeholders[r], tf.int32), num_or_size_splits=3, axis=1)
        e1v = tf.transpose(tf.squeeze(tf.gather(emtEmbed, e1, name='e1v' + str(r)), [1]))
        e2v = tf.transpose(tf.squeeze(tf.gather(emtEmbed, e2, name='e2v' + str(r)), [1]))
        e3v = tf.transpose(tf.squeeze(tf.gather(emtEmbed, e3, name='e3v' + str(r)), [1]))
        e1v_pos = e1v
        e2v_pos = e2v
        e1v_neg = e1v
        e2v_neg = e3v
        num_rel_r = tf.expand_dims(tf.shape(e1v_pos)[1], 0)
        preactivation_pos = list()
        preactivation_neg = list()

        print("e1v_pos: " + str(e1v_pos.get_shape()))
        print("W[r][:,:,slice]: " + str(W[r][:, :, 0].get_shape()))
        print("e2v_pos: " + str(e2v_pos.get_shape()))

        print("Starting preactivation funcs")
        for slice in range(k):
            preactivation_pos.append(tf.reduce_sum(e1v_pos * tf.matmul(W[r][:, :, slice], e2v_pos), 0))
            preactivation_neg.append(tf.reduce_sum(e1v_neg * tf.matmul(W[r][:, :, slice], e2v_neg), 0))

        preactivation_pos = tf.stack(preactivation_pos)
        preactivation_neg = tf.stack(preactivation_neg)

        temp2_pos = tf.matmul(V[r], tf.concat([e1v_pos, e2v_pos], 0))
        temp2_neg = tf.matmul(V[r], tf.concat([e1v_neg, e2v_neg], 0))

        print("   temp2_pos: " + str(temp2_pos.get_shape()))
        preactivation_pos = preactivation_pos + temp2_pos + b[r]
        preactivation_neg = preactivation_neg + temp2_neg + b[r]

        print("Starting activation funcs")
        activation_pos = tf.tanh(preactivation_pos)
        activation_neg = tf.tanh(preactivation_neg)

        score_pos = tf.reshape(tf.matmul(U[r], activation_pos), num_rel_r)
        score_neg = tf.reshape(tf.matmul(U[r], activation_neg), num_rel_r)
        print("score_pos: " + str(score_pos.get_shape()))
        if not is_eval:
            predictions.append(tf.stack([score_pos, score_neg]))
        else:
            predictions.append(tf.stack([score_pos, tf.reshape(label_placeholders[r], num_rel_r)]))
            print("score_pos_and_neg: " + str(predictions[r].get_shape()))

    print("Concating predictions")
    predictions = tf.concat(predictions, 1)
    print("predictions: {}".format(predictions.get_shape()))

    return predictions


def loss(predictions, regularization):
    print("Beginning building loss")
    temp1 = tf.maximum(tf.subtract(predictions[1, :], predictions[0, :]) + 1, 0)
    temp1 = tf.reduce_sum(temp1)
    temp2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))
    temp = temp1 + (regularization * temp2)
    print("loss: {}".format(temp))
    return temp


def training(loss, learningRate):
    print("Beginning building training")
    return tf.train.AdagradOptimizer(learningRate).minimize(loss)


def eval(predictions):
    print("predictions " + str(predictions.get_shape()))  # (batch_size * corrupt_size, 2)
    prediction, labels = tf.split(predictions, num_or_size_splits=2, axis=0)  # (batch_size * corrupt_size, 1)
    prediction = tf.concat((1 - tf.transpose(prediction)), tf.transpose(prediction))
    labels = ((tf.cast(tf.squeeze(tf.transpose(labels)), tf.int32)) + 1) / 2
    print("predictions " + str(prediction.get_shape()))
    print("labels " + str(labels.get_shape()))
    # get number of correct labels for the logits (if prediction is top 1 closest to actual)
    correct = tf.nn.in_top_k(prediction, labels, 1)
    # cast tensor to int and return number of correct labels
    return tf.reduce_sum(tf.cast(correct, tf.int32))

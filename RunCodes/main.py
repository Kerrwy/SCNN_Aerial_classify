from keras.models import load_model
import numpy as np
import os
import sys
import time
from spiking_model import Initial_spiking_model, set_time, reset
start = time.time()

filepath = '../config'
num_classes = 8

cnn_modelpath = os.path.join(os.getcwd(), 'VGG16Model_NewData_parsed.h5')
cnn_model = load_model(cnn_modelpath)
weights = cnn_model.get_weights()

scnn_model, num_timesteps, batch_size, dataset_path = Initial_spiking_model(filepath)
scnn_model.build(cnn_model, weights)

x_test = np.load(os.path.join(dataset_path, 'x_test.npz'))['arr_0']
y_test = np.load(os.path.join(dataset_path, 'y_test.npz'))['arr_0']
num_to_test = len(y_test)
num_batches = num_to_test/batch_size

truth_d = []
guesses_d = []
score_ann = 0
path_accuracy = os.path.join(os.path.pardir, 'accuracy.txt')
if os.path.exists(path_accuracy):
    os.remove(path_accuracy)


for batch_idx in range(num_batches):

    batch_idxs = range(batch_size * batch_idx, batch_size * (batch_idx + 1))
    x_b_l = x_test[batch_idxs, :]
    y_b_l = y_test[batch_idxs, :]
    truth_b = np.argmax(y_b_l, axis=1)
    output_b_l_t = np.zeros((batch_size, num_classes, num_timesteps))

    print("\nStarting new simulation...\n")
    print("Current accuracy of batch:")

    for ts in range(num_timesteps):
        sim_step = (ts + 1)
        set_time(scnn_model, sim_step)

        out_spikes = scnn_model.snn.predict_on_batch(x_b_l)
        output_b_l_t[:, :, ts] = out_spikes

        spike_sums_b_l = np.sum(output_b_l_t, 2)
        guesses_b = np.argmax(spike_sums_b_l, 1)

        from spiking_model import echo
        echo('{:.2%}_'.format(np.mean((truth_b) == guesses_b)))

    output_b_l_t = np.cumsum(output_b_l_t, 2)
    guesses_b_t = np.argmax(output_b_l_t, 1)

    top1snn_acc = np.mean(np.array(truth_b) == np.array(guesses_b_t[:, -1]))
    num_samples_seen = (batch_idx + 1) * batch_size


    print("\nBatch {} of {} completed ({:.1%})".format(
        batch_idx + 1, num_batches, (batch_idx + 1) / float(num_batches)))
    print("Moving accuracy of SNN : {:.2%}.".format(top1snn_acc))


    # Evaluate ANN on the same batch as SNN for a direct comparison.
    cnn_model.set_weights(weights)
    score = cnn_model.test_on_batch(x_b_l, y_b_l)
    score_ann += score[1] * batch_size

    top1ann_acc = score_ann / num_samples_seen
    print("Moving accuracy of ANN: {:.2%}."
          "\n".format(top1ann_acc))

    with open(path_accuracy, str('a')) as f_acc:
        f_acc.write(str("scnn, batch={}, acc={:.2%} \n".format(
            num_samples_seen, top1snn_acc)))
        f_acc.write(str("ann , batch={}, acc={:.2%} \n\n".format(num_samples_seen, top1ann_acc)))



    # Add results of current batch to previous results.
    truth_d += list(truth_b)
    guesses_d += list(guesses_b_t[:, -1])
    reset(scnn_model, batch_idx)

top1acc_total = np.mean(np.array(truth_d) == np.array(guesses_d))
# Print final result.
print("Simulation finished.\n\n")
print("Total accuracy: {:.2%} on {} test sample{}.\n".format(
    top1acc_total, len(guesses_d), num_to_test))
print("Total spend {} s".format(time.time()-start))

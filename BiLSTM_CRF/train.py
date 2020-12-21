import torch
import torch.nn as nn
import torch.optim as optim
import time
from util import DataUtil as dutil
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

torch.manual_seed(1)


#####################################################################
# Helper functions to make the code more readable.


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# 得到 sequence 的索引表示
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


#####################################################################
# Create model


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        # [seq_len,batch_size,embedding_size]
        # 句子中的单词个数，长度相等，包括 EOS
        # 有几个句子
        # 标签个数
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,  # 取整除 - 返回商的整数部分（向下取整）
                            num_layers=1, bidirectional=True)
        '''
        input_size ：输入的维度
        hidden_size：h的维度
        num_layers：堆叠LSTM的层数，默认值为1
        bias：偏置 ，默认值：True
        batch_first： 如果是True，则input为(batch, seq, input_size)。默认值为：False（seq_len, batch, input_size）
        bidirectional ：是否双向传播，默认值为False
        '''

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


#####################################################################
# Run training

def precision(tag_to_ix, output, target):
    total = 0
    correct = 0
    i = 0
    while i < len(output):
        if output[i] == tag_to_ix['S']:
            total += 1
            correct += 1 if target[i] == tag_to_ix['S'] else 0
        elif output[i] == tag_to_ix['B']:
            j = i
            i += 1
            while i < len(output) and output[i] == tag_to_ix['I']:
                i += 1
            if i < len(output) and output[i] == tag_to_ix['E']:
                total += 1
                correct += 1 if target[j:i + 1] == output[j:i + 1] else 0
            else:
                i -= 1
        i += 1
    return correct, total


START_TAG = "<START>"
STOP_TAG = "<STOP>"
# TODO
EMBEDDING_DIMs = [24, 32, 50]
HIDDEN_DIMs = [24, 32, 50]

if __name__ == '__main__':
    # TODO
    word_to_ix, data = dutil.BiLSTM_stat(['../DATASET/dataset1/train.utf8', '../DATASET/dataset2/train.utf8'])

    size = len(data) // 20

    train_set = data[:int(size * 0.9)]
    test_set = data[int(size * 0.9):size]

    print(size, len(train_set), len(test_set))
    # TODO
    T = 50

    tag_to_ix = {'B': 0, 'I': 1, 'E': 2, 'S': 3, START_TAG: 4, STOP_TAG: 5}
    models = []
    optimizers = []
    save_paths = []

    for EMBEDDING_DIM in EMBEDDING_DIMs:
        for HIDDEN_DIM in HIDDEN_DIMs:
            model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
            optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
            models.append(model)
            optimizers.append(optimizer)
            path = os.path.join('record',
                                'e' + str(EMBEDDING_DIM)
                                + 'h' + str(HIDDEN_DIM))
            """
            if not os.path.isdir(path):
                os.makedirs(path)
            """
            save_paths.append(path)
    """
    accuracys = np.zeros((len(models), T), dtype=float)

    for m in range(len(models)):
        model = models[m]
        optimizer = optimizers[m]
        # Check predictions before training
        # with torch.no_grad():
        #     precheck_sent = prepare_sequence(test_set[0][0], word_to_ix)
        #     precheck_tags = torch.tensor([tag_to_ix[t] for t in test_set[0][1]], dtype=torch.long)
        #     print(model(precheck_sent))

        # Make sure prepare_sequence from earlier in the LSTM section is loaded
        for epoch in range(T):  # again, normally you would NOT do 300 epochs, it is toy data
            start = time.time()
            for sentence, tags in train_set:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

                # Step 3. Run our forward pass.
                loss = model.neg_log_likelihood(sentence_in, targets)

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                optimizer.step()

            ###################### 保存模型 ########################
            # TODO
            save_path = os.path.join(save_paths[m], str(epoch + 1) + '.pt')
            torch.save(model.state_dict(), save_path)
            model.load_state_dict(torch.load(save_path))
            model.eval()
            #######################################################

            ###################### 测试部分 ########################
            correct = np.zeros(len(test_set), dtype=int)
            total = np.zeros(len(test_set), dtype=int)
            for t in range(len(test_set)):
                sentence = test_set[t][0]
                # tags = torch.tensor([tag_to_ix[i] for i in test_set[t][1]], dtype=torch.long)
                tags = [tag_to_ix[i] for i in test_set[t][1]]
                precheck_sent = prepare_sequence(sentence, word_to_ix)
                output = model(precheck_sent)
                correct[t], total[t] = precision(tag_to_ix, output[1], tags)
            accuracy = 1.0 * np.sum(correct) / np.sum(total) if np.sum(total) != 0 else 0.0
            accuracys[m][epoch] = accuracy
            #######################################################

            end = time.time()
            print(epoch + 1, ', time:', (end - start), ', accuracy:', accuracy)

        # Check predictions after training
        # with torch.no_grad():
        #     precheck_sent = prepare_sequence(test_set[0][0], word_to_ix)
        #     print(model(precheck_sent))
        # We got it!

    print(accuracys)
    # TODO
    pickle.dump(accuracys, open('record/accuracy.pickle', 'wb'))
    """
    accuracys = pickle.load(open('record/accuracy.pickle', 'rb'))

    ####################### 测试部分 #######################
    # 创建画板
    fig = plt.figure()

    # 创建画纸
    ax1 = fig.add_subplot(1, 1, 1)

    # test result
    ax1.set_title('Accuracy/Epoch')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    for m in range(len(accuracys)):
        ax1.plot(range(1, T + 1, 1), accuracys[m], '-',
                 label=os.path.splitext(os.path.basename(save_paths[m]))[0])
    plt.legend()
    plt.show()
    #######################################################

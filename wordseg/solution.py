from typing import List


class Solution:
    # --------------------
    # 在此填写 学号 和 用户名
    # --------------------
    ID = "17302010079"
    NAME = "宋怡景"

    # --------------------
    # 对于下方的预测接口，需要实现对你模型的调用：
    #
    # 要求：
    #    输入：一组句子
    #    输出：一组预测标签
    #
    # 例如：
    #    输入： ["我爱北京天安门", "今天天气怎么样"]
    #    输出： ["SSBEBIE", "BEBEBIE"]
    # --------------------

    # --------------------
    # 一个样例模型的预测
    # --------------------
    def example_predict(self, sentences: List[str]) -> List[str]:
        from .example_model import ExampleModel

        model = ExampleModel()
        results = []
        for sent in sentences:
            results.append(model.predict(sent))
        return results

    # --------------------
    # HMM 模型的预测接口
    # --------------------
    def hmm_predict(self, sentences: List[str]) -> List[str]:
        from HMM import HMM
        results = []
        hmm = HMM.HMMModel(['./DATASET/dataset1/train.utf8', './DATASET/dataset2/train.utf8'])
        for sent in sentences:
            results.append(''.join(hmm.viterbi(sent)))
        return results

    # --------------------
    # CRF 模型的预测接口
    # --------------------
    def crf_predict(self, sentences: List[str]) -> List[str]:
        from CRF import CRF
        results = []
        crf = CRF.CRFModel('./CRF/record/template4/normal/67.pickle', ['./DATASET/templates/template4.utf8'])
        for sent in sentences:
            results.append(''.join(crf.viterbi(sent)))
        return results

    # --------------------
    # DNN 模型的预测接口
    # --------------------
    def dnn_predict(self, sentences: List[str]) -> List[str]:
        from BiLSTM_CRF import BiLSTM_CRF
        from util import DataUtil as dutil
        word_to_ix, _ = dutil.BiLSTM_stat(['./DATASET/dataset1/train.utf8', './DATASET/dataset2/train.utf8'])
        results = []
        for sent in sentences:
            results.append(''.join(BiLSTM_CRF.viterbi(word_to_ix, './BiLSTM_CRF/final/e50h32/10.pt', sent)))
        return results

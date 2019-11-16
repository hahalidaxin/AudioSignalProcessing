import pickle
import re
from collections import Counter
from talk.mfcc import mfcc
from scipy.io import wavfile
from hmmlearn import hmm
import numpy as np
from scipy.special import logsumexp


class Speech:
    ''' 
    Speech
    --------------
    语料的包装类
    '''

    def __init__(self, dirName, fileName):
        self.fileName = fileName    # file name
        self.dirName = dirName
        self.features = None    # feature matrix
        self.soundSamplerate, self.sound = wavfile.read(dirName + fileName)
        
        pattern = re.compile(r'(\d+)_(\d+)_(\d+).wav')
        self.personId, self.categoryId, self.idx = pattern.match(self.fileName).group(1,2,3)

    def extractFeature(self):
        ''' mfcc feature extraction '''
        self.features = mfcc(self.sound, nwin=int(
            self.soundSamplerate * 0.03), fs=self.soundSamplerate, nceps=24)[0]


class SpeechRecognizer:
    ''' 
    SpeechRecognizer
    ----------------------
    HMM模型的包装类
    '''

    def __init__(self, categoryId):
        self.categoryId = categoryId
        self.trainData = []
        self.flagDataStacked = False
        self.trainDataLengths = None
        self.hmmModel = None

        self.nComp = 5  # number of states
        self.n_iter = 10    # number of iterations

    def initModelParam(self, nComp, n_iter):
        ''' 初始化 Gaussian HMM Model的参数 '''
        self.nComp = nComp  # number of states
        self.n_iter = n_iter    # number of iterations

    def stackTrainData(self):
        if self.flagDataStacked:
            return
        self.trainDataLengths = [x.shape[0] for x in self.trainData]
        # 对数组进行堆叠[[frames * [mfccwidth*1]],...] => [[frames x mfccwidth],...]
        self.trainData = np.vstack(self.trainData)
        self.flagDataStacked = True


    def saveHmmModel(self):
        with open("models/md_{0}.pkl".format(self.categoryId), "wb") as file: 
            pickle.dump(self.hmmModel, file)

    def loadHmmModel(self):
        with open("models/md_{0}.pkl".format(self.categoryId), "rb") as file: 
            self.hmmModel = pickle.load(file)

    def initHmmModel(self,load_model=False):
        '''
        initHmmModel
        -----------------
        初始化Hmm模型，如果load_model为真则加载已有模型
        '''
        if load_model:
            self.loadHmmModel()
        else:
            self.getHmmModel()

    def getHmmModel(self):
        ''' 进行Gaussian Hmm模型的初始化 '''

        # Gaussian HMM
        # nMix  GMM 总的状态数目
        # transmat_prior 转移概率矩阵的先验
        # startprob_prior 开始转移概率的先验
        # convariance_type HMM 中使用的协方差矩阵的类型
        #   默认‘diag’代表对于每一个状态都是用一个对角协方差矩阵
        # n_iter 进行 WM 算法迭代的次数
        model = hmm.GaussianHMM(n_components=self.nComp, n_iter=self.n_iter)
        self.hmmModel = model
        self.hmmModel._init(self.trainData, self.trainDataLengths)

    def trainHmmModel(self):
        ''' 训练Gaussian Hmm 模型 '''
        self.hmmModel.init_params = ''
        self.hmmModel.fit(self.trainData, self.trainDataLengths)


    def viterbi(self):
        '''
            使用 viterbi 算法推测当前训练数据及模型情况下最可能的隐藏序列
            获得隐藏序列之后通过简单统计方法估计模型参数
        '''
        state_seq_list = []
        for (i, j) in iter_from_X_lengths(self.trainData, self.trainDataLengths):
            framelogprob = self.hmmModel._compute_log_likelihood(
                self.trainData[i:j])
            logprob, state_sequence = self.hmmModel._do_viterbi_pass(
                framelogprob)
            state_seq_list.append(state_sequence)
        # 利用 state_seq_list 计算新的模型参数

        # 计算 startprob
        accum_map = Counter([x[0] for x in state_seq_list])
        self.hmmModel.startprob_ = np.array([accum_map[x]/len(self.trainDataLengths) for x in range(self.nComp)], dtype=np.float64)
        acc_transmat = np.zeros((self.nComp, self.nComp))
        for rki, state_seq in enumerate(state_seq_list):
            for rkj, state in enumerate(state_seq[1:]):
                pre_state = state_seq[rkj-1]
                acc_transmat[pre_state][state] = acc_transmat[pre_state][state] + 1
        self.hmmModel.transmat_ = acc_transmat / np.sum(acc_transmat, axis=1)[:, None]
        
def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {:d} samples in lengths array {!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]

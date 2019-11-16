import re
import os
import pickle
from utils import Speech, SpeechRecognizer

CATEGORY = ['0', '1', '2', '3', '4', '5', '6']  # 7 categories
test_set = {'1', '5', '9'}
# 注意 这里没有使用 1 3 录音人的语料
block_person_set = ['1', '3']

def loadData(dirName, train_flag = True):
    ''' 读取dirName下的所有数据，并且直接进行wav数据的mfcc特征提取 '''
    fileList = [f for f in os.listdir(dirName) if os.path.splitext(f)[1] == '.wav']
    fileList.sort()    
    speechList = []
    
    for fileName in fileList:
        pattern = re.compile(r'(\d+)_(\d+)_(\d+).wav')
        personId, categoryId, idx = pattern.match(fileName).group(1,2,3)

        if personId in block_person_set:
            continue

        if (train_flag and (not idx in test_set)) \
          or ((not train_flag) and (idx in test_set)):
            # print(fileName)
            speech = Speech(dirName, fileName)
            speech.extractFeature()
            speechList.append(speech)

    return speechList

def training(speechRecognizerList):
    ''' HMM training
    > 运行Baum-Welch算法进行模型的参数重估 '''
    for speechRecognizer in speechRecognizerList:
        speechRecognizer.trainHmmModel()

    return speechRecognizerList

def loadModel():
    print('loading all models now ...')
    speechRecognizerList = []
    
    # initialize speechRecognizer
    for categoryId in CATEGORY:
        speechRecognizer = SpeechRecognizer(categoryId)
        speechRecognizerList.append(speechRecognizer)
        speechRecognizer.initHmmModel(load_model=True)

    return speechRecognizerList

def saveModel(speechRecognizerList):
    for recognizer in speechRecognizerList:
        recognizer.saveHmmModel()

def pre_training(n_iter, speechList):
    ''' HMM pre training using viterbi
    > 首先对Gaussion HMM模型进行初始化，之后迭代运行viterbi算法对模型的初始参数进行确定'''
    speechRecognizerList = []
    
    # initialize speechRecognizer
    for categoryId in CATEGORY:
        speechRecognizer = SpeechRecognizer(categoryId)
        speechRecognizerList.append(speechRecognizer)
    
    # organize data into the same category
    for speechRecognizer in speechRecognizerList:
        # 针对一个语料使用 viterbi 算法进行 pre_training
        for speech in speechList:
            if speech.categoryId ==  speechRecognizer.categoryId:
                speechRecognizer.trainData.append(speech.features)
        
        # get hmm model
        # 当 nComp 设置不恰当的时候，运行 hmm 的过程中会出现nan的错误
        speechRecognizer.initModelParam(nComp = 12, n_iter = 15)
        speechRecognizer.stackTrainData()
        speechRecognizer.initHmmModel()

    for speechRecognizer in speechRecognizerList:
        for iter in range(n_iter):
            speechRecognizer.viterbi()

    return speechRecognizerList

def recognize(testSpeechList, speechRecognizerList):
    ''' recognition ''' 
    predictCategoryIdList = []
    
    for testSpeech in testSpeechList:
        scores = []
        
        for recognizer in speechRecognizerList:
            score = recognizer.hmmModel.score(testSpeech.features)
            scores.append(score)
        
        idx = scores.index(max(scores))
        predictCategoryId = speechRecognizerList[idx].categoryId
        predictCategoryIdList.append(predictCategoryId)

    return predictCategoryIdList


def calculateRecognitionRate(groundTruthCategoryIdList, predictCategoryIdList):
    ''' calculate recognition rate '''
    score = 0
    length = len(groundTruthCategoryIdList)
    
    for i in range(length):
        gt = groundTruthCategoryIdList[i]
        pr = predictCategoryIdList[i]
        
        if gt == pr:
            score += 1
    
    recognitionRate = float(score) / length
    return recognitionRate
    

def train4Models():
    ### Step.1 读取训练数据
    print('Step.1 Training data loading...')
    trainDir = 'audioRes/'
    trainSpeechList = loadData(trainDir, train_flag=True)
    print('d1one!') 
    ### Step.2 训练
    print('Step.2 Pre Training model...')
    speechRecognizerList = pre_training(7, trainSpeechList)

    print('Step.3 Training model...')
    speechRecognizerList = training(speechRecognizerList)
    print('done!')
    ### Step.3.5 Save Models 
    print('Step.3.5 Save All Models')
    saveModel(speechRecognizerList)
    ### Step.3 读取测试数据
    print('Step.3 Test data loading...')
    testDir = 'audioRes/'
    testSpeechList = loadData(testDir, train_flag=False)
    print('done!')
    ### Step.4 识别
    print('Step.4 Recognizing...')
    predictCategoryIdList = recognize(testSpeechList, speechRecognizerList)
    ### Step.5 打印结果
    groundTruthCategoryIdList = [speech.categoryId for speech in testSpeechList]
    recognitionRate = calculateRecognitionRate(groundTruthCategoryIdList, predictCategoryIdList)
    
    print('===== Final result =====')
    print('Ground Truth:\t', groundTruthCategoryIdList)
    print('Prediction:\t', predictCategoryIdList)
    print('Accuracy:\t', recognitionRate)
    

def testModels():
    ### Step.3.5 Save Models 
    print('Step.3.5 Save All Models')
    speechRecognizerList = loadModel()
    ### Step.3 读取测试数据
    print('Step.3 Test data loading...')
    testDir = 'audioRes/'
    testSpeechList = loadData(testDir, train_flag=False)
    print('done!')
    ### Step.4 识别
    print('Step.4 Recognizing...')
    predictCategoryIdList = recognize(testSpeechList, speechRecognizerList)
    ### Step.5 打印结果
    groundTruthCategoryIdList = [speech.categoryId for speech in testSpeechList]
    recognitionRate = calculateRecognitionRate(groundTruthCategoryIdList, predictCategoryIdList)
    
    print('===== Final result =====')
    print('Ground Truth:\t', groundTruthCategoryIdList)
    print('Prediction:\t', predictCategoryIdList)
    print('Accuracy:\t', recognitionRate)

def loadLabelName():
    labelNameList = {}
    with open('label_name.txt','r') as f:
        for line in f.readlines():
            x, y = line.split()
            labelNameList[x] = y
    return labelNameList

def inference(recognizerList, dirName, fileName, labelNameList):
    '''
    Params
    ----------
    recognizerList: recognizer 序列, 需要提前调用loadModel进行加载
    dirName: wav文件目录地址
    fileName: 需要识别的wav文件名称
    
    返回值：(预测得到的类别Id, 对应的类别语音label)

    Description
    ------------
    对fileName指定的wav文件进行mfcc特征提取后使用HmmModel进行推测，并返回推测类别ID
    '''
    print('inference wav file {0}'.format(fileName))
    speech = Speech(dirName, fileName)
    speech.extractFeature()

    scores = []
        
    for recognizer in recognizerList:
        score = recognizer.hmmModel.score(speech.features)
        scores.append(score)
    
    idx = scores.index(max(scores))
    predictCategoryId = recognizerList[idx].categoryId
    print('\tpredict result : {0}'.format(labelNameList[predictCategoryId]))
    return predictCategoryId, labelNameList[predictCategoryId]


if __name__ == '__main__':
    # 这里是一个调用接口的例子

    # 首先加载已经训练好的模型 
    recognizerList = loadModel()
    # 加载 labelName
    labelNameList = loadLabelName()
    # 利用HmmModel进行推理，得出预测结果
    inference(recognizerList, './audioRes/', '5_2_3.wav', labelNameList)
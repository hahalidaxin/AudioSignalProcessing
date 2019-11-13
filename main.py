import os
from utils import Speech, SpeechRecognizer

CATEGORY = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']  # 10 categories


def loadData(dirName):
    ''' 读取dirName下的所有数据，并且直接进行wav数据的mfcc特征提取 '''
    fileList = [f for f in os.listdir(dirName) if os.path.splitext(f)[1] == '.wav']
    
    speechList = []
    
    for fileName in fileList:
        speech = Speech(dirName, fileName)
        speech.extractFeature()
        
        speechList.append(speech)
    return speechList

def training(speechRecognizerList):
    ''' HMM training
    > 运行Baum-Welch算法进行模型的参数重估 '''
    # speechRecognizerList = []
    # # 
    # # initialize speechRecognizer
    # for categoryId in CATEGORY:
    #     speechRecognizer = SpeechRecognizer(categoryId)
    #     speechRecognizerList.append(speechRecognizer)
    
    # # organize data into the same category
    # for speechRecognizer in speechRecognizerList:
    #     for speech in speechList:
    #         if speech.categoryId ==  speechRecognizer.categoryId:
    #             speechRecognizer.trainData.append(speech.features)
        
    #     # get hmm model
    #     speechRecognizer.initModelParam(nComp = 5, nMix = 2, \
    #                                     covarianceType = 'diag', n_iter = 100, \
    #                                     bakisLevel = 2)
    #     speechRecognizer.stackTrainData()
    #     speechRecognizer.getHmmModel()

    for speechRecognizer in speechRecognizerList:
        speechRecognizer.trainHmmModel()

    return speechRecognizerList

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
        speechRecognizer.getHmmModel()

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
    

def main():
    ### Step.1 读取训练数据
    print('Step.1 Training data loading...')
    trainDir = '/projects/vison_audio/lab3/HMM/training_data/'
    trainSpeechList = loadData(trainDir)
    print('d1one!') 
    ### Step.2 训练
    print('Step.2 Pre Training model...')
    speechRecognizerList = pre_training(7, trainSpeechList)

    print('Step.3 Training model...')
    speechRecognizerList = training(speechRecognizerList)
    print('done!')
    ### Step.3 读取测试数据
    print('Step.3 Test data loading...')
    testDir = '/projects/vison_audio/lab3/HMM/test_data/'
    testSpeechList = loadData(testDir)
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
    

if __name__ == '__main__':
    main()
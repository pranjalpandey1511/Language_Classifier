import pprint

import numpy as np
import math
import sys
from re import search
import pickle
import random

# global Variables

features_name = ["containsQ", "AvgWordCount", "englishWords", "dutchWords", "vowelsCount", "vowelsConsonantratio",
                 "containsDoubleVowel", "containsEn", "containsIj", "someCommonDutchLetter", "Language"]

commonDutchWordSet = ["de", "hebben", "het", "van", "een", "meest", "niet"]
commonEnglishWordSet = ["the", "and", "of", "he", "have", "has"]
vowels = ["a", "i", "e", "o", "u"]
someCommonDutchLetter = ["ss", "gg", "kk", "pp", "aan", "rr", "nn", "oi", "op"]
defaultLanguage = "en"
avoidZero = 0.000000000001

booleanValueList = ["True", "False"]

fLableDictionary = {}
fLableDictionary[0] = "containsQ"
fLableDictionary[1] = "AvgWordCount"
fLableDictionary[2] = "englishWords"
fLableDictionary[3] = "dutchWords"
fLableDictionary[4] = "vowelsCount"
fLableDictionary[5] = "vowelsConsonentratio"
fLableDictionary[6] = "containsdoubleAEO"
fLableDictionary[7] = "containsEn"
fLableDictionary[8] = "containsIj"
fLableDictionary[9] = "containsPP"


def takeInput():
    inputParameters = []
    input1 = sys.argv[1]
    inputParameters.append(input1)
    if input1 == "train":
        examples = sys.argv[2]
        hypothesisOut = sys.argv[3]
        learningType = sys.argv[4]
        inputParameters.append(examples)
        inputParameters.append(hypothesisOut)
        inputParameters.append(learningType)
    elif input1 == "predict":
        hypothesis = sys.argv[2]
        file = sys.argv[3]
        inputParameters.append(hypothesis)
        inputParameters.append(file)

    return inputParameters


def featuresExtactTrain(examples):
    """
             This function is check features of each line and store it
             as boolean data according to the features present and used to
             Train
             :param examples: contains file to be read
             :return numpyFeatures: return the matrix of conditional data
    """
    global defaultLanguage
    enCount = 0
    nlCount = 0
    featureList = []
    for i in range(0, 11):
        featureList.append(False)
    numpyFeatures = np.array(featureList)
    with open(examples, encoding="utf-8") as file:
        count = 0
        for line in file:
            if line[:2] != 'en' and line[:2] != 'nl':
                continue
            if line[:2] == "en":
                enCount = enCount + 1
            else:
                nlCount = nlCount + 1
            featureList[10] = line[:2]
            line = line[3:]
            line = line.lower().replace(',', '')
            featureList[0] = checkFeatures(0, line)
            featureList[1] = checkFeatures(1, line)
            featureList[2] = checkFeatures(2, line)
            featureList[3] = checkFeatures(3, line)
            featureList[4] = checkFeatures(4, line)
            featureList[5] = checkFeatures(5, line)
            featureList[6] = checkFeatures(6, line)
            featureList[7] = checkFeatures(7, line)
            featureList[8] = checkFeatures(8, line)
            featureList[9] = checkFeatures(9, line)
            demoArray = np.array([featureList])
            numpyFeatures = np.vstack((numpyFeatures, demoArray))

    numpyFeatures = np.delete(numpyFeatures, (0), axis=0)

    if nlCount > enCount:
        defaultLanguage = "nl"
    return numpyFeatures


def featuresExtactPredict(examples):
    """
             This function is check features of each line and store it
             as boolean data according to the features present and used to
             predict
             :param examples: contains file to be read
             :return numpyFeatures: return the matrix of conditional data
    """
    featureList = []
    for i in range(0, 11):
        featureList.append(False)
    numpyFeatures = np.array(featureList)
    with open(examples, encoding="utf-8") as file:
        for line in file:
            featureList[10] = line[:2]
            line = line.lower()
            featureList[0] = checkFeatures(0, line)
            featureList[1] = checkFeatures(1, line)
            featureList[2] = checkFeatures(2, line)
            featureList[3] = checkFeatures(3, line)
            featureList[4] = checkFeatures(4, line)
            featureList[5] = checkFeatures(5, line)
            featureList[6] = checkFeatures(6, line)
            featureList[7] = checkFeatures(7, line)
            featureList[8] = checkFeatures(8, line)
            featureList[9] = checkFeatures(9, line)
            demoArray = np.array([featureList])
            numpyFeatures = np.vstack((numpyFeatures, demoArray))

    numpyFeatures = np.delete(numpyFeatures, (0), axis=0)
    numpyFeatures = np.delete(numpyFeatures, (10), axis=1)
    return numpyFeatures


def checkFeatures(count, line):
    """
             This function is check features of each line and return true/false according
             to the criteria
             :param count: for if else condition
             :param line: contains each lines read from textfile
             :return: boolean True if nl / False if en
    """
    line = line.strip("\n").strip(".").strip("!").strip("'")
    line = line.split(" ")
    check = True
    if count == 0:
        for word in line:
            if search("q", word):
                check = False
        return check

    if count == 1:
        wordsLengthCount = 0
        for word in line:
            wordsLengthCount = wordsLengthCount + len(word)
        avgWordLength = wordsLengthCount / len(line)
        if avgWordLength > 7:
            return True
        else:
            return False

    if count == 2:
        check = True
        for word in line:
            for engword in commonEnglishWordSet:
                if word == engword:
                    check = False
        return check

    if count == 3:
        check = False
        for word in line:
            for dutchword in commonDutchWordSet:
                if word == dutchword:
                    check = True
        return check

    if count == 4:
        vowelsCount = 0
        for word in line:
            for letter in word:
                if letter.lower() in vowels:
                    vowelsCount = vowelsCount + 1
        if vowelsCount < 45:
            return False
        else:
            return True

    if count == 5:
        consonentCount = 0
        vowelsCount = 0
        for word in line:
            for letter in word:
                if letter == " ":
                    continue
                if letter.lower() in vowels:
                    vowelsCount = vowelsCount + 1
                else:
                    consonentCount = consonentCount + 1
        if consonentCount != 0:
            ratio = vowelsCount / consonentCount
            if ratio >= 0.40 and ratio <= 0.55:
                return False
            elif ratio > 0.62:
                return True
            else:
                return False
        else:
            return True

    if count == 6:
        countd = 0
        for word in line:
            if search("ee", word) or search("oo", word) or search("aa", word) or search("uu", word) or search("ii",
                                                                                                              word):
                countd = countd + 1

        if countd > 1:
            return True
        else:
            return False

    if count == 7:
        enCount = 0
        for word in line:
            if search("en", word):
                enCount = enCount + 1

        if enCount >= 1:
            return True
        else:
            return False

    if count == 8:
        countIJ = 0
        for word in line:
            if search("ij", word):
                countIJ = countIJ + 1

        if countIJ > 1:
            return True
        else:
            return False

    if count == 9:
        countdouble = 0
        for word in line:
            for commonletter in someCommonDutchLetter:
                if search(commonletter, word):
                    countdouble = countdouble + 1
        if countdouble >= 1:
            return True
        else:
            return False


def entropy(features, data, target):
    """
             This function is to calculate the information gain entropy for each feature
             :param data: data condition Matrix extracted from lines
             :param features: contains feature labels
             :param target: the target attribute used to make decision tree
            :return dataEntropy: returns the calculated Entropy
        """
    targetIndex = features.index(target)
    numberDict = {}
    dataEntropy = 0.0
    if data.size == 1:
        return 0

    for row in data:
        val = row[targetIndex]
        if val in numberDict:
            numberDict[val] = numberDict[val] + 1
        else:
            numberDict[val] = 1

    totalValues = numberDict.values()

    for number in totalValues:
        dataEntropy += (-number / len(data)) * math.log2(number / len(data))

    return dataEntropy


def informationGain(attributes, data, attr, target):
    """
         This function is to calculate the information gain for each feature
         :param data: data condition Matrix extracted from lines
         :param attributes: contains feature labels
         :param attr: current attribute in consideration to calculate information gain
         :param target: the target attribute used to make decision tree
        :return infoGain: returns the information gain value
    """
    subEntropy = 0.0
    infoGain = 0.0
    targetIndex = attributes.index(attr)

    valueSet = data[:, targetIndex]
    countTrue = 0
    countFalse = 0
    for value in valueSet:
        if str(value) == str(True):
            countTrue = countTrue + 1
        else:
            countFalse = countFalse + 1
    pTrue = countTrue / len(valueSet)
    pFalse = countFalse / len(valueSet)

    demolist = []
    a = []
    for i in range(len(attributes)):
        demolist.append(True)

    a.append(demolist)
    datasubsetTrue = np.array(a)
    dataSubsetFalse = np.array(a)

    for x in data:
        if str(x[targetIndex]) == str(True):
            datasubsetTrue = np.vstack((datasubsetTrue, x))
        else:
            dataSubsetFalse = np.vstack((dataSubsetFalse, x))

    datasubsetTrue = np.delete(datasubsetTrue, (0), axis=0)
    dataSubsetFalse = np.delete(dataSubsetFalse, 0, axis=0)
    subEntropy = pTrue * entropy(attributes, datasubsetTrue, target) + pFalse * entropy(attributes, dataSubsetFalse,
                                                                                        target)
    parentEntropy = entropy(attributes, data, target)
    infoGain = parentEntropy - subEntropy

    return infoGain


def chooseBestFeature(data, features, target):
    """
         This function is to get the select the best dataset based on the information gain
         :param data: data condition Matrix extracted from lines
         :param features: contains feature labels
         :param target: the target attribute used to make decision tree
        :return best: returns the best feature
    """
    best = features[0]
    maxGain = 0

    for feature in features[:-1]:  # [:-1] to except label attribute
        tempGain = informationGain(features, data, feature, target)
        if tempGain > maxGain:  # iterate to find best attribute which has maximum information gain
            maxGain = tempGain
            best = feature

    return best


def getNewData(textData, features, bestAttribute, value):
    """
         This function is to get the new dataset
         :param textData: data condition Matrix extracted from lines
         :param features: contains feature labels
         :param bestAttribute: the best attribute used to make decision tree
         :param value: True or False
        :return tree: the new data
    """
    v = []
    for x in textData:
        v = x
        break
    newData = np.array([v])

    bestFeatureIndex = features.index(bestAttribute)

    for row in textData:
        if str(row[bestFeatureIndex]) == str(value):
            newData = np.vstack((newData, row))

    newData = np.delete(newData, (0), axis=0)
    newData = np.delete(newData, bestFeatureIndex, axis=1)
    # newData = np.unique(newData, axis=0)
    return newData


def calculate_purality(data, targetIndex):
    """
        This function is used to train the data, it will make decision tree
        for the prediction of the data
        :param data: updated condition Matrix extracted from lines
        :param targetIndex: Index of the target Attribute
        :return tree: return the maximum count of label present in data
    """
    value = data[:, targetIndex]
    countEn = 0
    countNl = 0
    for x in value:
        if str(x) == str("nl"):
            countNl = countNl + 1
        else:
            countEn = countEn + 1

    if countEn > countNl:
        return "en"
    else:
        return "nl"


def makeDt(features, textData, targetAttribute, default):
    """
        This function is used to train the data, it will make decision tree
        for the prediction of the data
        :param features: contains feature labels
        :param textData: updated condition Matrix extracted from lines
        :param targetAttribute: contains the stumps the target attribute Language
        :param default: contains the default language i.e english
        :return tree: return the final decision tree
    """
    treeDict = {}
    targetIndex = features.index(targetAttribute)

    for row in textData:
        val = row[targetIndex]
        if val in treeDict:
            treeDict[val] = treeDict[val] + 1
        else:
            treeDict[val] = 1

    if len(features) <= 1 or textData.size == 0:
        return calculate_purality(textData, targetIndex)
    elif len(treeDict.keys()) == 1:
        return list(treeDict.keys())[0]
    else:
        bestAttribute = chooseBestFeature(textData, features, targetAttribute)
        tree = {bestAttribute: {}}

        # given best feature, make tree recursively with new dataset and featureset
        for value in booleanValueList:
            newData = getNewData(textData, features, bestAttribute, value)
            newAttr = features[:]
            newAttr.remove(bestAttribute)
            subtree = makeDt(newAttr, newData, targetAttribute, default)
            tree[bestAttribute][value] = subtree

        return tree


def predict(attributes, tree, default, data):
    """
        This function is used to predict the data, it will use decision tree
        to predict the data
        :param attributes: contains feature labels
        :param tree: contains the final decision tree
        :param default: contains the default language based on the counts
        :param data: updated condition Matrix extracted from lines
    """

    testSet = data
    resultSet = []

    count = 1
    for testTuple in testSet:
        test = testTuple
        next = tree

        while True:
            # if there is no more child node stop
            if not isinstance(next, dict):
                break
            attr = list(next.keys())
            attr = attr[0]
            index = attributes.index(attr)
            next = next[attr]
            if test[index] in next:
                next = next[test[index]]
            else:
                next = default
                break
        resultSet.append(next)
        count = count + 1

    result = ""
    for op in resultSet:
        result += op + '\n'

    print(result)


def adaTrain(features, featureArray, targerAttribute, defaultLanguage, noOfIteration):
    """
        This function is used to boost train the data, it will be give the desired
        stumps according to the noOfIteration
        :param features: contains feature labels
        :param featureArray: updated condition Matrix extracted from lines
        :param targetAttribute: contains the stumps the target attribute Language
        :param defaultLanguage: contains the default language i.e english
        :param noOfIteration: contains the max iterations which defines max stumps
        :return returnStumpList: return the stump with its significance
    """
    data = featureArray
    initialWeight = 1 / len(featureArray)
    initialWeightList = []
    for i in range(len(featureArray)):
        initialWeightList.append(initialWeight)
    demoArray = np.array(initialWeightList)
    data = np.column_stack((data, demoArray))
    bestFeature = ""
    stumpList = []
    returnStumpList = []
    significanceList = []
    featuresCopy = features[:]
    for i in range(0, noOfIteration):
        stump, bestFeature = chooseBestStump(featuresCopy, np.delete(data, len(featuresCopy), axis=1), targerAttribute,
                                             defaultLanguage)
        bestFeatureIndex = featuresCopy.index(bestFeature)
        significance = calculateSignificance(featuresCopy, data, bestFeature, targerAttribute)
        stumpList.append(stump)
        significanceList.append(significance)

        weakClassifierData, data = updateWeights(significance, data, featuresCopy, bestFeature, targerAttribute)
        data = boostWeakClassfier(weakClassifierData, data)
        data = np.delete(data, bestFeatureIndex, axis=1)
        featuresCopy.remove(bestFeature)

    for i in range(0, len(stumpList)):
        ss = []
        ss.append(stumpList[i])
        ss.append(significanceList[i])
        returnStumpList.append(ss)

    return returnStumpList


def boostWeakClassfier(weakData, data):
    """
         This function is to randomly boost the week data so for the next iteration
         it will be boosted and taken into consideration
         :param data: data condition Matrix extracted from lines with its sample weight
         :param weakData: ata which are weakly/incorrect classified
         :return nwData: data where weak classifiers are boosted
    """
    randomSize = random.uniform(0.3, 0.5)
    rowCount = int(randomSize * len(data)) - len(weakData)
    if rowCount < 0 or len(weakData) == 0:
        return data
    nwData = weakData
    for row in weakData:
        nwData = np.vstack((nwData, row))
        if len(nwData) == rowCount:
            break

    for row in data:
        nwData = np.vstack((nwData, row))
        if len(nwData) == len(data):
            break

    return nwData


def chooseBestStump(features, textData, targetAttribute, default):
    """
         This function is to select the best stump
         :param textData: data condition Matrix extracted from lines
         :param features: contains feature labels
         :param targetAttribute: contains the stumps the target attribute Language
         :param default: default prediction i.e english
         :return tree: the stump
         :return bestAttribute: the best attribute used to make a stump of max depth = 1
    """
    targetIndex = features.index(targetAttribute)
    bestAttribute = chooseBestFeature(textData, features, targetAttribute)
    bestFeatureIndex = features.index(bestAttribute)
    tree = {bestAttribute: {}}
    countTrueEn = 0
    countFalseEn = 0
    countTrueNl = 0
    countFalseNl = 0
    trueFinalVal = ""
    falseFinalVal = ""
    for row in textData:
        targetValue = row[targetIndex]
        bestFeatureValue = row[bestFeatureIndex]
        if str(bestFeatureValue) == str(True) and str(targetValue) == str("en"):
            countTrueEn = countTrueEn + 1
        if str(bestFeatureValue) == str(True) and str(targetValue) == str("nl"):
            countTrueNl = countTrueNl + 1
        if str(bestFeatureValue) == str(False) and str(targetValue) == str("en"):
            countFalseEn = countFalseEn + 1
        if str(bestFeatureValue) == str(False) and str(targetValue) == str("nl"):
            countFalseNl = countFalseNl + 1

    if countTrueNl > countTrueEn:
        trueFinalVal = "nl"
    else:
        trueFinalVal = "en"

    if countFalseNl > countFalseEn:
        falseFinalVal = "nl"
    else:
        falseFinalVal = "en"

    if trueFinalVal == falseFinalVal:
        trueFinalVal = "nl"
        falseFinalVal = "en"

    tree[bestAttribute][booleanValueList[0]] = trueFinalVal
    tree[bestAttribute][booleanValueList[1]] = falseFinalVal

    return tree, bestAttribute


def calculateSignificance(features, data, bestAttribute, targetAttribute):
    """
         This function is to calculate the significance with the formula
         :param data: data condition Matrix extracted from lines with its sample weight
         :param features: contains feature labels
         :param bestAttribute: contains attribute/feature which is the best stump
         :param targetAttribute: contains the stumps the target attribute Language
         :return significance: return the calculated alpha/significance/ weightHypothesis
    """
    global avoidZero
    totalValue = len(data)
    bestIndex = features.index(bestAttribute)
    targetIndex = features.index(targetAttribute)
    incorrectValues = 0
    for row in data:
        value = row[bestIndex]
        targetValue = row[targetIndex]
        if str(value) == str(True) and str(targetValue) == str("en"):
            incorrectValues = incorrectValues + 1
        elif str(value) == str(False) and str(targetValue) == str("nl"):
            incorrectValues = incorrectValues + 1

    totalError = (incorrectValues / totalValue)
    if totalError == 0:
        return 0
    calC = (1 - totalError) / totalError
    significance = 0.5 * np.log(calC)
    return significance


def updateWeights(wHypothesis, data, features, bestAttribute, targetAttribute):
    """
        This function is used to update the latest weight and then return the data with normalised
        weights updated on it
        :param wHypothesis: the significance
        :param data: data condition Matrix extracted from lines with its sample weight
        :param features: contains feature labels
        :param bestAttribute: contains attribute/feature which is the best stump
        :param targetAttribute: contains the stumps the target attribute Language
        :return data: updated condition Matrix extracted from lines with normalised weights
        :return weakClassifiers: data which are weakly/incorrect classified
    """
    weightCorrect = math.exp(-wHypothesis)
    weightIncorrect = math.exp(wHypothesis)
    weightColumnIndex = len(features)
    totalValue = len(data)
    bestIndex = features.index(bestAttribute)
    targetIndex = features.index(targetAttribute)
    incorrectValues = 0
    demoList = features[:]
    demoList.append("Weights")
    demoArray = []
    demoArray.append(demoList)
    weakClassifiers = np.array(demoArray)

    for row in data:
        value = row[bestIndex]
        targetValue = row[targetIndex]
        if str(value) == str(True) and str(targetValue) == str("en"):
            incorrectValues = incorrectValues + 1
            row[weightColumnIndex] = float(row[weightColumnIndex]) * weightIncorrect
            weakClassifiers = np.vstack((weakClassifiers, row))
        elif str(value) == str(False) and str(targetValue) == str("nl"):
            incorrectValues = incorrectValues + 1
            row[weightColumnIndex] = float(row[weightColumnIndex]) * weightIncorrect
            weakClassifiers = np.vstack((weakClassifiers, row))
        else:
            row[weightColumnIndex] = float(row[weightColumnIndex]) * weightCorrect

    weightSampleList = data[:, weightColumnIndex]
    sumWeight = 0
    for weights in weightSampleList:
        sumWeight = sumWeight + float(weights)

    for row in data:
        row[weightColumnIndex] = float(row[weightColumnIndex]) / sumWeight

    weakClassifiers = np.delete(weakClassifiers, (0), axis=0)
    if weakClassifiers.size != 0:
        for row in weakClassifiers:
            row[weightColumnIndex] = float(row[weightColumnIndex]) / sumWeight

    return weakClassifiers, data


def adaPredict(stumpList, features, data):
    """
        This function is used to predict the adaboost by using decision tree classifier
        and print the final classification
        :param stumpList: contains the stumps with its significance
        :param features: contains feature labels
        :param data condition Matrix extracted from lines
    """
    stumpFeatureList = []
    signiList = []
    for tuple in stumpList:
        for item in tuple:
            if isinstance(item, dict):
                for key in item:
                    stumpFeatureList.append(key)
            else:
                ft = float(item)
                signiList.append(ft)
    for row in data:
        sum = 0
        for i in range(0, len(stumpFeatureList)):
            indexinFeature = features.index(stumpFeatureList[i])
            signValue = signiList[i]
            val = row[indexinFeature]
            if str(val) == str(True):
                sum = sum + signValue
            else:
                sum = sum - signValue
        if sum > 0:
            print("nl")
        else:
            print("en")


if __name__ == '__main__':
    """
    main function:
    Print the output, generate the hypothesis file, train/predict
    according to the input
    """
    # contains the list of input
    inputParam = takeInput()
    features = []
    for x in features_name:
        features.append(x)
    # Train the data
    if len(inputParam) > 3:
        featureArray = featuresExtactTrain(inputParam[1])
        Tree = {}
        if inputParam[3] == "dt":
            Tree = makeDt(features, featureArray, "Language", defaultLanguage)
            # pprint.pprint(Tree)
        else:
            maxIteration = 5
            Tree = adaTrain(features, featureArray, "Language", defaultLanguage, maxIteration)

        hypothesisFile = open(inputParam[2] + ".pk", 'wb')
        pickle.dump(Tree, hypothesisFile)
        hypothesisFile.close()

    # predict the data
    else:
        f_myfile = open(inputParam[1] + ".pk", 'rb')
        objectData = pickle.load(f_myfile)
        predictFeatureArray = featuresExtactPredict(inputParam[2])
        # print(type(objectData))
        if str(type(objectData)) == "<class 'dict'>":
            predict(features, objectData, defaultLanguage, predictFeatureArray)
        else:
            adaPredict(objectData, features, predictFeatureArray)

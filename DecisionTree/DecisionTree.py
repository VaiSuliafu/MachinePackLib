# dependency packages
from math import log

'''
Represents a decision tree machine learning object.
'''
class DecisionTree:



    # constructor
    ## default measure will be entropy unless specified otherwise
    def __init__(DTself, measure = "entropy", maxDepth = 999):
        DTself.measure = measure
        DTself.maxDepth = maxDepth

    # define some helper methods
    '''
    Passes a header to the DecisionTree so that it may be printed.
    '''
    def setHeader(DTself, header):
        DTself.header = header
        return

    '''
    Returns a set of unique values in the argument oclumn col.
    '''
    def getUniqueClassSet(DTself, rows, col):
        return set([row[col] for row in rows])

    '''
    Returns a dictionary of class labels and their observed count in
    the target value column.
    '''
    def getClassCount(DTself, rows):

        countDic = {} # initialize empty dictionary
        for row in rows:
            key = row[-1] #iterate through rows in col
            if (key not in countDic): #if not yet in dictionary
                countDic[key] = 0 # add it to dictionary
            countDic[key] += 1 # increment the value stored with this class key
        return countDic

    '''
    Determines if the value is numeric.
    '''
    def isNumeric(DTself, value):
        return isinstance(value, int) or isinstance(value, float)


    '''
    Creates a Decision object.
    '''
    def createDecision(DTself, column, value):

        return DTself.Decision(DTself, column, value)

    '''
    Decision sub class.
    '''
    class Decision:
        '''
        Used at each node to partition the data.
    
        Records a column number index and an observed value.
        '''
    
        # constructor
        def __init__(Dself, decisionTree, column, value):
            Dself.column = column
            Dself.value = value
            Dself.decisionTree = decisionTree
        
            return

        # give Decision access to DT's isNumeric() method
        def isNumeric(Dself, value):

            # call this decision's DecisionTree.isNumeric()
            return Dself.decisionTree.isNumeric(value)
        
        # compares the sample value to the value
        # of this decision
        def compare(Dself, sample):
            s_val = sample[Dself.column]
        
            # if numeric, compare to this.value
            if (Dself.decisionTree.isNumeric(s_val)):
                return s_val >= Dself.value
            else:
                return s_val == Dself.value
            return
    
        def uniqueSet(Dself, sample):
            return Dself.decisionTree.getUniqueClassSet(sample, Dself.column)
        
        def getValue(Dself, sample):
            return sample[Dself.column]
    
        def __repr__(Dself):
            # Prints this decision
            condition = "=="
        
            if Dself.isNumeric(Dself.value):
                condition = ">="
        
            return "%s %s %s?" % (Dself.decisionTree.header[Dself.column], condition, str(Dself.value))

    '''
    Iterates through each row in rows and uses the
    decision.compare to split the row into boolean
    subgroups.
    '''
    def split(DTself, rows, decision):
        
#         # dictionary of unique labels
#         groups = {}
        
#         # store list as value under the label key
#         for label in decision.uniqueSet():
#             groups[label] = []
            
#         for row in rows:
#             groups[decision.getValue(row)].append(row)
            
        # initialize the buckets
        trueGroup, falseGroup = [], []
        
        # loop through rows argument
        for row in rows:
            if decision.compare(row):
                trueGroup.append(row)
            else:
                falseGroup.append(row)
    
        # return buckets
        return trueGroup, falseGroup
        # return groups

    # now to define some different functions
# for measuring information gain

    '''
    Calculates the entropy for this subset of data.
    '''
    def entropy(DTself, rows):
    
        # calculate the count of each value in target label
        counts = DTself.getClassCount(rows)
        entropy_result = 0
    
        # first calculate proportion of each value in target label
        for classes in counts:

            cp = counts[classes] / float(len(rows))
        
            # now calculate entropy using the proportions
            entropy_result +=  (-cp) * (log(cp) / log(2))
        
        return entropy_result

    '''
    Calculates the majority error for this subset of data.
    '''
    def majorityError(DTself, rows):

        # calculate the count of each value in target label
        counts = DTself.getClassCount(rows)

        # get max class count
        maxClass = max(counts, key = counts.get)

        # return majority error
        return (len(rows) -  counts[maxClass]) / len(rows)

    '''
    Calculates the Gini impurity for this subset of data.
    '''
    def gini(DTself, rows):
    
        # calculate the count of each class in subset
        counts = DTself.getClassCount(rows)
        impurity = 1
    
        # calculate the proportion of each class label
        for classes in counts:
        
            class_proportion = counts[classes] / float(len(rows))
        
            # now calculate the impurity
            impurity = impurity - class_proportion**2
    
        return impurity

    '''
    The uncertainty of the starting node, minus the
    weighted impurity of two child nodes.
    '''
    def informationGain(DTself, left, right, uncertainty):
    
        # calculate the cardinality of left branch
        p = float(len(left)) / (len(left) + len(right))
    
        # decide which uncertainty calculation to return
        if (DTself.measure == "entropy"):
            # uncertarinty
            lGain = (p * DTself.entropy(left))
            rGain = ((1 - p) * DTself.entropy(right))
            iGain = uncertainty - (lGain + rGain)

        if (DTself.measure == "gini"):
            # uncertainty
            lGain = (p * DTself.gini(left))
            rGain = ((1 - p) * DTself.gini(right))
            iGain = uncertainty - (lGain + rGain)

        if (DTself.measure == "majorityerror"):
            # uncertainty
            lGain = (p * DTself.majorityError(left))
            rGain = ((1 - p) * DTself.majorityError(right))
            iGain = uncertainty - (lGain + rGain)

        return iGain, lGain, rGain

    '''
    Returns the best information gain and decision.

    Loops through each column and class combination,
    calculating the hypothetical information gain
    associated with each possible decision.'''
    def findBestSplit(DTself, rows):
    
        # variable which will store the best info gain
        maxGain = 0;
    
        # will store best decision to return later
        maxDecision = None

        # will store the individual info gain of left and right nodes respectively
        maxLeftGain = 0
        maxRightGain = 0
    
        # calculate the uncertainty at this node
        if (DTself.measure == "entropy"):

            currentUncertainty = DTself.entropy(rows)

        elif (DTself.measure == "gini"):

            currentUncertainty = DTself.gini(rows)

        elif (DTself.measure == "majorityerror"):

            currentUncertainty = DTself.majorityError(rows)
    
        # calculate number of predictor columns
        colCount = len(rows[0]) - 1
    
        # loop through the columns
        for col in range(colCount):
        
            # get count dictionary of unique classes
            uniqueLabels = set([row[col] for row in rows])
        
            # loop through each unique label
            for label in uniqueLabels:
            
                # initiate the decision object
                decision = DTself.createDecision(col, label)
            
                # calculate potential split
                trueRows, falseRows = DTself.split(rows, decision)
            
                # if split is 0, purity is 100%
                if (len(trueRows) == 0 or len(falseRows) == 0):
                
                    continue;
                
                # calculate information gain from splitting
                infoGain, infoGainLeft, infoGainRight = DTself.informationGain(trueRows, falseRows, currentUncertainty)
                
#                 print("")
#                 print("Decision {}".format(decision))
#                 print("IG {}".format(infoGain))
#                 print("IG_L {}".format(infoGainLeft))
#                 print("IG_R {}".format(infoGainRight))
            
                if (infoGain >= maxGain):
                    maxGain = infoGain
                    maxDecision = decision
                    maxLeftGain = infoGainLeft
                    maxRightGain = infoGainRight
                
        return maxGain, maxDecision, currentUncertainty, maxLeftGain, maxRightGain

    '''
    Method to create a Leaf.
    '''
    def createLeaf(DTself, rows):
        return DTself.Leaf(DTself, rows)

    '''
    Leaf sub class.
    '''
    class Leaf:
        """
    A leaf class that holds information about the
    proportions of target variable values in this leaf's
    sub data rows. 

    These proportions are used as predictions.
        """
    
        # constructor
        def __init__(Lself, decisionTree, rows):
            counts = decisionTree.getClassCount(rows)
            Lself.predictions = max(counts, key = counts.get)
            return

    '''
    Creates a DecisionNode.
    '''
    def createDecisionNode(DTself, decision, trueBranch, falseBranch, uncertainty, predictions, infoGain, infoGainLeft, infoGainRight):
        return DTself.DecisionNode(decision, trueBranch, falseBranch, uncertainty, predictions, infoGain, infoGainLeft, infoGainRight)

    '''
    DecisionNode sub class.
    '''
    class DecisionNode:
        """
        The object which hold a decision and pointers
        to the true and false branches.
        """

        #constructor
        def __init__(DNself, decision, trueBranch, falseBranch, uncertainty, predictions, infoGain, infoGainLeft, infoGainRight):
        
            DNself.decision = decision
            DNself.trueBranch = trueBranch
            DNself.falseBranch = falseBranch
            DNself.uncertainty = uncertainty
            DNself.predictions = predictions
            DNself.infoGain = infoGain
            DNself.uncertaintyLeft = infoGainLeft
            DNself.uncertaintyRight = infoGainRight

            return

    """
    Builds the decision tree.

    Uses recursion to build the tree, where the base case
    is that there is no further information gain.
    """
    def buildTree(DTself, rows):
    
        # loop through the input data calculating
        # hypothetical information gains
        infoGain, decision, uncertainty, infoGainLeft, infoGainRight = DTself.findBestSplit(rows)
    
        # BASE CASE
        # return if no more informaton game
        # and make a leaf
        if (infoGain == 0):
            return DTself.createLeaf(rows)
    
        # else, split the data on this decision
        trueRows, falseRows = DTself.split(rows, decision)
    
        # recurse on the true data
        trueBranch = DTself.buildTree(trueRows)
    
        # recurse on the false data
        falseBranch = DTself.buildTree(falseRows)

        # get proportion
        predictions = DTself.getClassCount(rows)
    
        # return a DecisionNode object
        # which will point to the two
        # branches created and stored
        # in this function as well as the decision
        
        node = DTself.createDecisionNode(decision, trueBranch, falseBranch, uncertainty,
                                         predictions, infoGain, infoGainLeft, infoGainRight)

    
        # return leaf if maximum depth is reached
        if (DTself.getDepth(node) >= DTself.maxDepth):
            return DTself.createLeaf(rows)
        else:
            return node
    
    
    def printTree(DTself, node, spacing=""):
        '''
        Recursive function which prints a decision tree.
        '''
    
        # BASE CASE
        if isinstance(node, DTself.Leaf):
            #print("")
            print (spacing + "Predictions: ", node.predictions)
            return
    
        # print decision of this node
        #print("")

        print(spacing + "Decision: [{}]".format(node.decision))
        print(spacing + "Sample: {}".format(node.predictions))
        print(spacing + "Uncertainty: {0:.2f}".format(node.uncertainty))
        print(spacing + "Left Uncertainty: {0:.2f}".format(node.uncertaintyLeft))
        print(spacing + "Right Uncertainty: {0:.2f}".format(node.uncertaintyRight))
        print(spacing + "Info Gain: {0:.2f}".format(node.infoGain))
    
        # recurse on this nodes trueBranch
        print("")
        print(spacing + "----> True:")
        DTself.printTree(node.trueBranch, spacing + "  ")
    
        # recurse on this nodes falseBranch
        print("")
        print(spacing + "----> False:")
        DTself.printTree(node.falseBranch, spacing + "  ")
        
    def getDepth(DTself, node):
            
        # base case
        if isinstance(node, DTself.Leaf):
            return 1;
            
        # recursive
        tDepth = DTself.getDepth(node.trueBranch)
        fDepth = DTself.getDepth(node.falseBranch)

        return 1 + max(tDepth, fDepth)
    
    def predict(DTself, row, node):
        
        # get predictions if leaf node
        if isinstance(node, DTself.Leaf):
            return node.predictions
        
        # otherwise decide which branch to travel
        if node.decision.compare(row):
            return DTself.predict(row, node.trueBranch)
        else:
            return DTself.predict(row, node.falseBranch)
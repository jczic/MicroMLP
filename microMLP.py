"""
The MIT License (MIT)
Copyright © 2018 Jean-Christophe Bos & HC² (www.hc2.fr)
"""


from math import exp, log
from ujson import load, dumps
from utime import time

try :
    from machine import rng
except :
    from urandom import random

class MicroMLP :

    ACTFUNC_HEAVISIDE   = 'Heaviside'
    ACTFUNC_SIGMOID     = 'Sigmoid'
    ACTFUNC_TANH        = 'TanH'
    ACTFUNC_SOFTPLUS    = 'SoftPlus'
    ACTFUNC_RELU        = 'ReLU'
    ACTFUNC_GAUSSIAN    = 'Gaussian'

    Eta                 = 0.30
    Alpha               = 0.75
    Gain                = 0.99

    CorrectLearnedMAE   = 0.02

    # -------------------------------------------------------------------------
    # --( Class : NNValue )----------------------------------------------------
    # -------------------------------------------------------------------------

    class NNValue :

        # -[ Static functions ]---------------------------------

        @staticmethod
        def FromPercent(value) :
            return MicroMLP.NNValue(0, 100, value)
        @staticmethod
        def NewPercent() :
            return MicroMLP.NNValue.FromPercent(0)

        @staticmethod
        def FromByte(value) :
            return MicroMLP.NNValue(0, 255, ord(value))
        @staticmethod
        def NewByte() :
            return MicroMLP.NNValue.FromByte(b'\x00')

        @staticmethod
        def FromBool(value) :
            return MicroMLP.NNValue(0, 1, 1 if value else 0)
        @staticmethod
        def NewBool() :
            return MicroMLP.NNValue.FromBool(False)

        @staticmethod
        def FromAnalogSignal(value) :
            return MicroMLP.NNValue(0, 1, value)
        @staticmethod
        def NewAnalogSignal() :
            return MicroMLP.NNValue.FromAnalogSignal(0)

        # -[ Constructor ]--------------------------------------

        def __init__(self, minValue, maxValue, value) :
            if maxValue - minValue <= 0 :
                raise Exception('MicroMLP.NNValue : "maxValue" must be greater than "minValue".')
            self._minValue = minValue
            self._maxValue = maxValue
            self._value    = 0.0
            self._setScaledValue(minValue, maxValue, value)

        # -[ Private functions ]--------------------------------

        def _setScaledValue(self, minValue, maxValue, value) :
            if   value <= minValue : self._value = 0.0
            elif value >= maxValue : self._value = 1.0
            else                   : self._value = float(value - minValue) / (maxValue - minValue)

        # -[ Properties ]---------------------------------------

        @property
        def AsFloat(self) :
            return self._minValue + (self._value * (self._maxValue - self._minValue))
        @AsFloat.setter
        def AsFloat(self, value) :
            self._setScaledValue(self._minValue, self._maxValue, value)

        @property
        def AsInt(self) :
            return int(round(self.AsFloat))
        @AsInt.setter
        def AsInt(self, value) :
            self._setScaledValue(self._minValue, self._maxValue, value)

        @property
        def AsPercent(self) :
            return self._value * 100
        @AsPercent.setter
        def AsPercent(self, value) :
            self._setScaledValue(0, 100, value)

        @property
        def AsByte(self) :
            return chr(int(round(self._value * 255)))
        @AsByte.setter
        def AsByte(self, value) :
            self._setScaledValue(0, 255, ord(value))

        @property
        def AsBool(self) :
            return self._value >= 0.5
        @AsBool.setter
        def AsBool(self, value) :
            self._setScaledValue(0, 1, 1 if value else 0)

        @property
        def AsAnalogSignal(self) :
            return self._value
        @AsAnalogSignal.setter
        def AsAnalogSignal(self, value) :
            self._setScaledValue(0, 1, value)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # --( Class : Connection )-------------------------------------------------
    # -------------------------------------------------------------------------

    class Connection :

        # -[ Constructor ]--------------------------------------

        def __init__(self, neuronSrc, neuronDst, weight=None) :
            neuronSrc.AddOutputConnection(self)
            neuronDst.AddInputConnection(self)
            self._neuronSrc           = neuronSrc
            self._neuronDst           = neuronDst
            self._weight              = weight if weight else MicroMLP.RandomNetworkWeight()
            self._momentumDeltaWeight = 0.0

        # -[ Public functions ]---------------------------------

        def UpdateWeight(self, eta, alpha) :
            deltaWeight                = eta \
                                       * self._neuronSrc.ComputedOutput \
                                       * self._neuronDst.ComputedSignalError
            self._weight              += deltaWeight + (alpha * self._momentumDeltaWeight)
            self._momentumDeltaWeight  = deltaWeight

        def Remove(self) :
            if self._neuronSrc and self._neuronDst :
                nSrc = self._neuronSrc
                nDst = self._neuronDst
                self._neuronSrc = None
                self._neuronDst = None
                nSrc.RemoveOutputConnection(self)
                nDst.RemoveInputConnection(self)

        # -[ Properties ]---------------------------------------

        @property
        def NeuronSrc(self) :
            return self._neuronSrc

        @property
        def NeuronDst(self) :
            return self._neuronDst

        @property
        def Weight(self) :
            return self._weight

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # --( Class : Neuron )-----------------------------------------------------
    # -------------------------------------------------------------------------

    class Neuron :

        # -[ Constructor ]--------------------------------------

        def __init__(self, parentLayer) :
            parentLayer.AddNeuron(self)
            self._parentLayer           = parentLayer
            self._inputConnections      = [ ]
            self._outputConnections     = [ ]
            self._inputBias             = None
            self._computedInput         = 0.0
            self._computedOutput        = 0.0
            self._computedDeltaError    = 0.0
            self._computedSignalError   = 0.0

        # -[ Public functions ]---------------------------------

        def GetNeuronIndex(self) :
            return self._parentLayer.GetNeuronIndex(self)

        def GetInputConnections(self) :
            return self._inputConnections

        def GetOutputConnections(self) :
            return self._outputConnections

        def AddInputConnection(self, connection) :
            self._inputConnections.append(connection)

        def AddOutputConnection(self, connection) :
            self._outputConnections.append(connection)

        def RemoveInputConnection(self, connection) :
            self._inputConnections.remove(connection)

        def RemoveOutputConnection(self, connection) :
            self._outputConnections.remove(connection)

        def SetBias(self, bias) :
            self._inputBias = bias

        def GetBias(self) :
            return self._inputBias

        def SetOutputNNValue(self, nnvalue) :
            self._computedOutput = nnvalue.AsAnalogSignal

        def _computeInput(self) :
            sum = 0.0
            for conn in self._inputConnections :
                sum += conn.NeuronSrc.ComputedOutput * conn.Weight
            if self._inputBias :
                sum += self._inputBias.Value * self._inputBias.Weight
            self._computedInput = sum

        def ComputeOutput(self) :
            self._computeInput()
            if self._parentLayer._actFunc :
                self._computedOutput = self._parentLayer._actFunc( self._computedInput * \
                                                                   self._parentLayer.ParentMicroMLP.Gain )

        def ComputeError(self, targetNNValue=None) :
            if targetNNValue :
                self._computedDeltaError = targetNNValue.AsAnalogSignal - self.ComputedOutput
            else :
                self._computedDeltaError = 0.0
                for conn in self._outputConnections :
                    self._computedDeltaError += conn.NeuronDst.ComputedSignalError * conn.Weight
            if self._parentLayer._actFunc :
                self._computedSignalError = self._computedDeltaError              \
                                          * self._parentLayer.ParentMicroMLP.Gain \
                                          * self._parentLayer._actFunc( self._computedInput,
                                                                        derivative = True )

        def Remove(self) :
            for conn in self._inputConnections :
                conn.NeuronSrc.RemoveOutputConnection(conn)
            for conn in self._outputConnections :
                conn.NeuronDst.RemoveInputConnection(conn)
            l = self._parentLayer
            self._parentLayer = None
            l.RemoveNeuron(self)

        # -[ Properties ]---------------------------------------

        @property
        def ParentLayer(self) :
            return self._parentLayer

        @property
        def ComputedOutput(self) :
            return self._computedOutput

        @property
        def ComputedDeltaError(self) :
            return self._computedDeltaError

        @property
        def ComputedSignalError(self) :
            return self._computedSignalError

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # --( Class : Bias )-------------------------------------------------------
    # -------------------------------------------------------------------------

    class Bias :

        # -[ Constructor ]--------------------------------------

        def __init__(self, neuronDst, value=1.0, weight=None) :
            neuronDst.SetBias(self)
            self._neuronDst           = neuronDst
            self._value               = value
            self._weight              = weight if weight else MicroMLP.RandomNetworkWeight()
            self._momentumDeltaWeight = 0.0

        # -[ Public functions ]---------------------------------

        def UpdateWeight(self, eta, alpha) :
            deltaWeight                = eta \
                                       * self._value \
                                       * self._neuronDst.ComputedSignalError
            self._weight              += deltaWeight + (alpha * self._momentumDeltaWeight)
            self._momentumDeltaWeight  = deltaWeight

        def Remove(self) :
            nDst.SetBias(None)

        # -[ Properties ]---------------------------------------

        @property
        def NeuronDst(self) :
            return self._neuronDst

        @property
        def Value(self) :
            return self._value

        @property
        def Weight(self) :
            return self._weight

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # --( Class : Layer )------------------------------------------------------
    # -------------------------------------------------------------------------

    class Layer :

        # -[ Constructor ]--------------------------------------

        def __init__(self, parentMicroMLP, activationFuncName=None, neuronsCount=0) :
            self._parentMicroMLP        = parentMicroMLP
            self._actFuncName           = activationFuncName
            self._actFunc               = MicroMLP.GetActivationFunction(activationFuncName)
            self._neurons               = [ ]
            self._parentMicroMLP.AddLayer(self)
            for i in range(neuronsCount) :
                MicroMLP.Neuron(self)

        # -[ Public functions ]---------------------------------

        def GetLayerIndex(self) :
            return self._parentMicroMLP.GetLayerIndex(self)

        def GetNeuron(self, neuronIndex) :
            if neuronIndex >= 0 and neuronIndex < len(self._neurons) :
                return self._neurons[neuronIndex]
            return None

        def GetNeuronIndex(self, neuron) :
            return self._neurons.index(neuron)

        def AddNeuron(self, neuron) :
            self._neurons.append(neuron)

        def RemoveNeuron(self, neuron) :
            self._neurons.remove(neuron)

        def GetMeanSquareError(self) :
            if len(self._neurons) == 0 :
                return 0
            mse = 0.0
            for n in self._neurons :
                mse += n.ComputedDeltaError ** 2
            return mse / len(self._neurons)

        def GetMeanAbsoluteError(self) :
            if len(self._neurons) == 0 :
                return 0
            mae = 0.0
            for n in self._neurons :
                mae += abs(n.ComputedDeltaError)
            return mae / len(self._neurons)

        def GetMeanSquareErrorAsPercent(self) :
            return round( self.GetMeanSquareError() * 100 * 1000 ) / 1000

        def GetMeanAbsoluteErrorAsPercent(self) :
            return round( self.GetMeanAbsoluteError() * 100 * 1000 ) / 1000

        def Remove(self) :
            while len(self._neurons) > 0 :
                self._neurons[0].Remove()
            mlp = self._parentMicroMLP
            self._parentMicroMLP = None
            mlp.RemoveLayer(self)

        # -[ Properties ]---------------------------------------

        @property
        def ParentMicroMLP(self) :
            return self._parentMicroMLP

        @property
        def ActivationFuncName(self) :
            return self._actFuncName

        @property
        def Neurons(self) :
            return self._neurons

        @property
        def NeuronsCount(self) :
            return len(self._neurons)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # --( Class : InputLayer )-------------------------------------------------
    # -------------------------------------------------------------------------

    class InputLayer(Layer) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, parentMicroMLP, neuronsCount=0) :
            super().__init__(parentMicroMLP, None, neuronsCount)

        # -[ Public functions ]---------------------------------

        def SetInputVectorNNValues(self, inputVectorNNValues) :
            if len(inputVectorNNValues) == self.NeuronsCount :
                for i in range(self.NeuronsCount) :
                    self._neurons[i].SetOutputNNValue(inputVectorNNValues[i])
                return True
            return False

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # --( Class : OutputLayer )------------------------------------------------
    # -------------------------------------------------------------------------

    class OutputLayer(Layer) :

        # -[ Constructor ]--------------------------------------

        def __init__(self, parentMicroMLP, activationFuncName, neuronsCount=0) :
            super().__init__(parentMicroMLP, activationFuncName, neuronsCount)

        # -[ Public functions ]---------------------------------

        def GetOutputVectorNNValues(self) :
            nnvalues = [ ]
            for n in self._neurons :
                nnvalues.append(MicroMLP.NNValue.FromAnalogSignal(n.ComputedOutput))
            return nnvalues

        def ComputeTargetLayerError(self, targetVectorNNValues) :
            if len(targetVectorNNValues) == self.NeuronsCount :
                for i in range(self.NeuronsCount) :
                    self._neurons[i].ComputeError(targetVectorNNValues[i])
                return True
            return False

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    # -[ Constructor ]--------------------------------------

    def __init__(self) :
        self._layers   = [ ]
        self._examples = [ ]

    # -[ Static functions ]-------------------------------------

    @staticmethod
    def Create(neuronsByLayers, activationFuncName, layersAutoConnectFunction=None, useBiasValue=1.0) :
        if not neuronsByLayers or len(neuronsByLayers) < 2 :
            raise Exception('MicroMLP.Create : Incorrect "neuronsByLayers" parameter.')
        for x in neuronsByLayers :
            if x < 1 :
                raise Exception('MicroMLP.Create : Incorrect count in "neuronsByLayers".')
        if not MicroMLP.GetActivationFunction(activationFuncName) :
            raise Exception('MicroMLP : Unknow activationFuncName "%s".' % activationFuncName)
        mlp = MicroMLP()
        for i in range(len(neuronsByLayers)) :
            if i == 0 :
                layer = MicroMLP.InputLayer(mlp, neuronsByLayers[i])
            else :
                if i == len(neuronsByLayers)-1 :
                    layer = MicroMLP.OutputLayer(mlp, activationFuncName, neuronsByLayers[i])
                else :
                    layer = MicroMLP.Layer(mlp, activationFuncName, neuronsByLayers[i])
                if layersAutoConnectFunction :
                    layersAutoConnectFunction(mlp.GetLayer(i-1), layer)
                if useBiasValue :
                    for n in layer.Neurons :
                        MicroMLP.Bias(n, useBiasValue)
        return mlp

    @staticmethod
    def RandomFloat() :
        if 'rng' in globals() :
            return rng() / (2 ** 24)
        return random()

    @staticmethod
    def RandomNetworkWeight() :
        return (MicroMLP.RandomFloat()-0.5) * 0.7

    @staticmethod
    def HeavisideActivation(x, derivative=False) :
        if derivative :
            return 1.0
        return 1.0 if x >= 0 else 0.0

    @staticmethod
    def SigmoidActivation(x, derivative=False) :
        f = 1.0 / ( 1.0 + exp(-x) )
        if derivative :
            return f * (1.0-f)
        return f

    @staticmethod
    def TanHActivation(x, derivative=False) :
        f = ( 2.0 / (1.0 + exp(-2.0 * x)) ) - 1.0
        if derivative :
            return 1.0 - (f ** 2)         
        return f

    @staticmethod
    def SoftPlusActivation(x, derivative=False) :
        if derivative :
            return 1 / (1 + exp(-x))
        return log(1 + exp(x))

    @staticmethod
    def ReLUActivation(x, derivative=False) :
        if derivative :
            return 1.0 if x >= 0 else 0.0
        return max(0.0, x)

    @staticmethod
    def GaussianActivation(x, derivative=False) :
        f = exp(-x ** 2)
        if derivative :
            return -2 * x * f
        return f

    @staticmethod
    def LayersFullConnect(layerSrc, layerDst) :
        if layerSrc and layerDst and layerSrc != layerDst :
            for nSrc in layerSrc.Neurons :
                for nDst in layerDst.Neurons :
                    MicroMLP.Connection(nSrc, nDst)

    @staticmethod
    def GetActivationFunction(actFuncName) :
        if actFuncName :
            funcs = {
                MicroMLP.ACTFUNC_HEAVISIDE : MicroMLP.HeavisideActivation,
                MicroMLP.ACTFUNC_SIGMOID   : MicroMLP.SigmoidActivation,
                MicroMLP.ACTFUNC_TANH      : MicroMLP.TanHActivation,
                MicroMLP.ACTFUNC_SOFTPLUS  : MicroMLP.SoftPlusActivation,
                MicroMLP.ACTFUNC_RELU      : MicroMLP.ReLUActivation,
                MicroMLP.ACTFUNC_GAUSSIAN  : MicroMLP.GaussianActivation
            }
            if actFuncName in funcs :
                return funcs[actFuncName]
        return None
    
    @staticmethod
    def LoadFromFile(filename) :
            with open(filename, 'r') as jsonFile :
                o = load(jsonFile)
            mlp       = MicroMLP()
            mlp.Eta   = o['Eta']
            mlp.Alpha = o['Alpha']
            mlp.Gain  = o['Gain']
            oLayers   = o['Layers']
            for i in range(len(oLayers)) :
                oLayer             = oLayers[i]
                activationFuncName = oLayer['Func']
                oNeurons           = oLayer['Neurons']
                if i == 0 :
                    layer = MicroMLP.InputLayer(mlp, len(oNeurons))
                else :
                    if i == len(oLayers)-1 :
                        layer = MicroMLP.OutputLayer(mlp, activationFuncName, len(oNeurons))
                    else :
                        layer = MicroMLP.Layer(mlp, activationFuncName, len(oNeurons))
                for neuron in layer.Neurons :
                    oNeuron = oNeurons[neuron.GetNeuronIndex()]
                    oBias   = oNeuron['Bias']
                    if oBias :
                        MicroMLP.Bias(neuron, oBias['Val'], oBias['Wght'])
                    for oConn in oNeuron['Conn'] :
                        nSrc = mlp.GetLayer(oConn['LSrc']).GetNeuron(oConn['NSrc'])
                        MicroMLP.Connection(nSrc, neuron, oConn['Wght'])
            return mlp

    # -[ Public functions ]---------------------------------

    def GetLayer(self, layerIndex) :
        if layerIndex >= 0 and layerIndex < len(self._layers) :
            return self._layers[layerIndex]
        return None

    def GetLayerIndex(self, layer) :
        return self._layers.index(layer)

    def AddLayer(self, layer) :
        self._layers.append(layer)

    def RemoveLayer(self, layer) :
        self._layers.remove(layer)

    def ClearAll(self) :
        while len(self._layers) > 0 :
            self._layers[0].Remove()

    def GetInputLayer(self) :
        if self.LayersCount > 0 :
            l = self._layers[0]
            if type(l) is MicroMLP.InputLayer :
                return l
        return None

    def GetOutputLayer(self) :
        if self.LayersCount > 0 :
            l = self._layers[self.LayersCount-1]
            if type(l) is MicroMLP.OutputLayer :
                return l
        return None

    def Learn(self, inputVectorNNValues, targetVectorNNValues) :
        if targetVectorNNValues :
            return self._simulate(inputVectorNNValues, targetVectorNNValues, True)
        return False

    def Test(self, inputVectorNNValues, targetVectorNNValues) :
        if targetVectorNNValues :
            return self._simulate(inputVectorNNValues, targetVectorNNValues)
        return False

    def Predict(self, inputVectorNNValues) :
        if self._simulate(inputVectorNNValues) :
            return self.GetOutputLayer().GetOutputVectorNNValues()
        return None

    def QLearningLearnForChosenAction( self,
                                       stateVectorNNValues,
                                       rewardNNValue,
                                       pastStateVectorNNValues,
                                       chosenActionIndex,
                                       terminalState           = True,
                                       discountFactorNNValue   = None ) :
        if chosenActionIndex >= 0 and \
           chosenActionIndex < self.GetOutputLayer().NeuronsCount :
            if not terminalState :
                if not discountFactorNNValue or \
                   not self._simulate(stateVectorNNValues) :
                    return False
                bestActVal = 0
                for nnVal in self.GetOutputLayer().GetOutputVectorNNValues() :
                    if nnVal.AsAnalogSignal > bestActVal :
                        bestActVal = nnVal.AsAnalogSignal
            if self._simulate(pastStateVectorNNValues) :
                targetVectorNNValues = self.GetOutputLayer().GetOutputVectorNNValues()
                targetActVal         = rewardNNValue.AsAnalogSignal
                if not terminalState :
                    targetActVal += discountFactorNNValue.AsAnalogSignal * bestActVal
                targetVectorNNValues[chosenActionIndex].AsAnalogSignal = targetActVal
                return self._simulate(pastStateVectorNNValues, targetVectorNNValues, True)
        return False

    def QLearningPredictBestActionIndex(self, stateVectorNNValues) :
        bestActIdx = None
        if self._simulate(stateVectorNNValues) :
            maxVal = 0
            idx    = 0
            for nnVal in self.GetOutputLayer().GetOutputVectorNNValues() :
                if nnVal.AsAnalogSignal > maxVal :
                    maxVal     = nnVal.AsAnalogSignal
                    bestActIdx = idx
                idx += 1
        return bestActIdx

    def SaveToFile(self, filename) :
        o = {
            'Eta'     : self.Eta,
            'Alpha'   : self.Alpha,
            'Gain'    : self.Gain,
            'Layers'  : [ ]
        }
        for layer in self.Layers :
            oLayer = {
                'Func'    : layer.ActivationFuncName,
                'Neurons' : [ ]
            }
            for neuron in layer.Neurons :
                bias = neuron.GetBias()
                if bias :
                    oBias = {
                        'Val'  : bias.Value,
                        'Wght' : bias.Weight
                    }
                else :
                    oBias = None
                oNeuron = {
                    'Bias' : oBias,
                    'Conn' : [ ]
                }
                for conn in neuron.GetInputConnections() :
                    oNeuron['Conn'].append( {
                        'LSrc' : conn.NeuronSrc.ParentLayer.GetLayerIndex(),
                        'NSrc' : conn.NeuronSrc.GetNeuronIndex(),
                        'Wght' : conn.Weight
                    } )
                oLayer['Neurons'].append(oNeuron)
            o['Layers'].append(oLayer)
        try :
            jsonStr  = dumps(o)
            jsonFile = open(filename, 'wt')
            jsonFile.write(jsonStr)
            jsonFile.close()
        except :
            return False
        return True

    def AddExample(self, inputVectorNNValues, targetVectorNNValues) :
        if self.IsNetworkComplete and \
           inputVectorNNValues    and \
           targetVectorNNValues   and \
           len(inputVectorNNValues)  == self.GetInputLayer().NeuronsCount and \
           len(targetVectorNNValues) == self.GetOutputLayer().NeuronsCount :
           self._examples.append( {
                'Input'  : inputVectorNNValues,
                'Target' : targetVectorNNValues
           } )
           return True
        return False

    def ClearExamples(self) :
        self._examples.clear()

    def LearnExamples(self, maxSeconds=30, maxCount=None, stopWhenLearned=True, printMAEAverage=True) :
        if self.ExamplesCount > 0 and maxSeconds > 0 :
            count   = 0
            endTime = time() + maxSeconds
            while time() < endTime and \
                  ( maxCount is None or count < maxCount ) :
                idx = int( MicroMLP.RandomFloat() * self.ExamplesCount )
                if not self.Learn( self._examples[idx]['Input'],
                                   self._examples[idx]['Target'] ) :
                    return 0
                count += 1
                if (stopWhenLearned or printMAEAverage) and count % 10 == 0 :
                    maeAvg = 0.0
                    for ex in self._examples :
                        self.Test(ex['Input'], ex['Target'])
                        maeAvg += self.MAE
                    maeAvg /= self.ExamplesCount
                    if printMAEAverage :
                        print( "[ STEP : %s / ERROR : %s%% ]"
                               % ( count, round(maeAvg*100*1000)/1000 ) )
                    if stopWhenLearned and maeAvg <= self.CorrectLearnedMAE :
                        break
            return count
        return 0

    # -[ Properties ]---------------------------------------

    @property
    def Layers(self) :
        return self._layers

    @property
    def LayersCount(self) :
        return len(self._layers)

    @property
    def IsNetworkComplete(self) :
        return self.GetInputLayer() is not None and self.GetOutputLayer() is not None

    @property
    def MSE(self) :
        if self.IsNetworkComplete :
            return self.GetOutputLayer().GetMeanSquareError()
        return 0.0

    @property
    def MAE(self) :
        if self.IsNetworkComplete :
            return self.GetOutputLayer().GetMeanAbsoluteError()
        return 0.0

    @property
    def MSEPercent(self) :
        if self.IsNetworkComplete :
            return self.GetOutputLayer().GetMeanSquareErrorAsPercent()
        return 0.0

    @property
    def MAEPercent(self) :
        if self.IsNetworkComplete :
            return self.GetOutputLayer().GetMeanAbsoluteErrorAsPercent()
        return 0.0

    @property
    def ExamplesCount(self) :
        return len(self._examples)

    # -[ Private functions ]------------------------------------

    def _propagateSignal(self) :
        if self.IsNetworkComplete :
            idx = 1
            while idx < self.LayersCount :
                for n in self.GetLayer(idx).Neurons :
                    n.ComputeOutput()
                idx += 1
            return True
        return False

    def _backPropagateError(self) :
        if self.IsNetworkComplete :
            idx = self.LayersCount-1
            while idx >= 0 :
                for n in self.GetLayer(idx).Neurons :
                    if idx < self.LayersCount-1 :
                        if idx > 0 :
                            n.ComputeError()
                        for conn in n.GetOutputConnections() :
                            conn.UpdateWeight(self.Eta, self.Alpha)
                    bias = n.GetBias()
                    if bias :
                        bias.UpdateWeight(self.Eta, self.Alpha)
                idx -= 1
            return True
        return False

    def _simulate(self, inputVectorNNValues, targetVectorNNValues=None, training=False) :
        if self.IsNetworkComplete and self.GetInputLayer().SetInputVectorNNValues(inputVectorNNValues) :
            self._propagateSignal()
            if not targetVectorNNValues :
                return not training
            if self.GetOutputLayer().ComputeTargetLayerError(targetVectorNNValues) :
                if not training :
                    return True
                return self._backPropagateError()
        return False

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

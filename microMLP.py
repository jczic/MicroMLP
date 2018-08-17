"""
The MIT License (MIT)
Copyright © 2018 Jean-Christophe Bos & HC² (www.hc2.fr)
"""


from math import pow, exp
from json import load, dumps
from time import time

try :
    from machine import rng
except :
    from random import random

class MicroMLP :

    ACTFUNC_BINARY   = 'Binary'
    ACTFUNC_SIGMOID  = 'Sigmoid'
    ACTFUNC_TANH     = 'Tanh'
    ACTFUNC_RELU     = 'ReLU'
    ACTFUNC_GAUSSIAN = 'Gaussian'

    Eta              = 0.30
    Alpha            = 0.75
    Gain             = 2.70

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
            self._neuronSrc      = neuronSrc
            self._neuronDst      = neuronDst
            self._weight         = weight if weight else MicroMLP.RandomFloat()
            self._momentumWeight = 0.0

        # -[ Public functions ]---------------------------------

        def ComputeWeight(self, eta, alpha) :
            self._weight         += ( eta   * self._neuronSrc.ComputedValue * self._neuronDst.ComputedError ) \
                                  + ( alpha * self._momentumWeight )
            self._momentumWeight  = ( eta   * self._neuronSrc.ComputedValue * self._neuronDst.ComputedError )

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

        def __init__(self, parentLayer, activateFunctionName) :
            parentLayer.AddNeuron(self)
            self._parentLayer           = parentLayer
            self._activateFunctionName  = activateFunctionName
            self._activateFunction      = MicroMLP.GetActivationFunction(activateFunctionName)
            self._inputConnections      = [ ]
            self._outputConnections     = [ ]
            self._computedValue         = 0.5
            self._computedError         = 0.0
            self._computedDeltaError    = 0.0

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

        def SetComputedNNValue(self, nnvalue) :
            self._computedValue = nnvalue.AsAnalogSignal

        def ComputeValue(self) :
            sum = 0.0
            for conn in self._inputConnections :
                sum += conn.NeuronSrc.ComputedValue * conn.Weight
            if self._activateFunction :
                self._computedValue = self._activateFunction(sum, self._parentLayer.ParentMicroMLP.Gain)

        def ApplyError(self, deltaError) :
            self._computedError      = self._parentLayer.ParentMicroMLP.Gain             \
                                     * self._computedValue * (1.0 - self._computedValue) \
                                     * deltaError;
            self._computedDeltaError = deltaError;

        def ComputeError(self, training=False) :
            deltaError = 0.0
            for conn in self._outputConnections :
                deltaError += conn.NeuronDst.ComputedError * conn.Weight
                if training :
                    conn.ComputeWeight( self._parentLayer.ParentMicroMLP.Eta,
                                        self._parentLayer.ParentMicroMLP.Alpha )
            self.ApplyError(deltaError)

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
        def ActivateFunctionName(self) :
            return self._activateFunctionName

        @property
        def ComputedValue(self) :
            return self._computedValue

        @property
        def ComputedError(self) :
            return self._computedError

        @property
        def ComputedDeltaError(self) :
            return self._computedDeltaError

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # --( Class : Layer )------------------------------------------------------
    # -------------------------------------------------------------------------

    class Layer :

        # -[ Constructor ]--------------------------------------

        def __init__(self, parentMicroMLP, neuronsCount=0, activateFunctionName=None) :
            self._parentMicroMLP        = parentMicroMLP
            self._activateFunctionName  = activateFunctionName
            self._neurons               = [ ]
            self._parentMicroMLP.AddLayer(self)
            for i in range(neuronsCount) :
                MicroMLP.Neuron(self, activateFunctionName)

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

        def ComputeLayerValues(self) :
            for n in self._neurons :
                n.ComputeValue()

        def ComputeLayerErrors(self, training=False) :
            for n in self._neurons :
                n.ComputeError(training)

        def GetMeanSquareError(self) :
            if len(self._neurons) == 0 :
                return 0
            mse = 0.0
            for n in self._neurons :
                mse += n.ComputedDeltaError * n.ComputedDeltaError
            return mse / len(self._neurons)

        def GetMeanAbsoluteError(self) :
            if len(self._neurons) == 0 :
                return 0
            mae = 0.0
            for n in self._neurons :
                mae += abs(n.ComputedDeltaError)
            return mae / len(self._neurons)

        def GetMeanSquareErrorAsPercent(self) :
            return self.GetMeanSquareError() * 100

        def GetMeanAbsoluteErrorAsPercent(self) :
            return self.GetMeanAbsoluteError() * 100

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
        def ActivateFunctionName(self) :
            return self._activateFunctionName

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
            super().__init__(parentMicroMLP, neuronsCount)

        # -[ Public functions ]---------------------------------

        def SetInputVectorNNValues(self, inputVectorNNValues) :
            if len(inputVectorNNValues) == self.NeuronsCount :
                for i in range(self.NeuronsCount) :
                    self._neurons[i].SetComputedNNValue(inputVectorNNValues[i])
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

        def __init__(self, parentMicroMLP, neuronsCount=0, activateFunctionName=None) :
            super().__init__(parentMicroMLP, neuronsCount, activateFunctionName)

        # -[ Public functions ]---------------------------------

        def GetOutputVectorValues(self) :
            nnvalues = [ ]
            for n in self._neurons :
                nnvalues.append(MicroMLP.NNValue.FromAnalogSignal(n.ComputedValue))
            return nnvalues

        def ComputeTargetLayerError(self, targetVectorNNValues) :
            if len(targetVectorNNValues) == self.NeuronsCount :
                for i in range(self.NeuronsCount) :
                    deltaError = targetVectorNNValues[i].AsAnalogSignal - self._neurons[i].ComputedValue
                    self._neurons[i].ApplyError(deltaError)
                return True
            return False

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    # -[ Constructor ]--------------------------------------

    def __init__(self, activateFunctionName) :
        if not MicroMLP.GetActivationFunction(activateFunctionName) :
            raise Exception('MicroMLP : Unknow activateFunctionName (%s).' % activateFunctionName)
        self._activateFunctionName = activateFunctionName
        self._layers               = [ ]
        self._examples             = [ ]

    # -[ Static functions ]-------------------------------------

    @staticmethod
    def Create(neuronsByLayers, activateFunctionName, layersAutoConnectFunction=None) :
        if not neuronsByLayers or len(neuronsByLayers) < 2 :
            raise Exception('MicroMLP.Create : Incorrect "neuronsByLayers" parameter.')
        for x in neuronsByLayers :
            if x < 1 :
                raise Exception('MicroMLP.Create : Incorrect count in "neuronsByLayers".')
        mlp = MicroMLP(activateFunctionName)
        newLayer  = None
        lastLayer = None
        for i in range(len(neuronsByLayers)) :
            if i == 0 :
                newLayer = MicroMLP.InputLayer(mlp, neuronsByLayers[i])
            elif i == len(neuronsByLayers)-1 :
                newLayer = MicroMLP.OutputLayer(mlp, neuronsByLayers[i], activateFunctionName)
            else :
                newLayer = MicroMLP.Layer(mlp, neuronsByLayers[i], activateFunctionName)
            if layersAutoConnectFunction and lastLayer :
                layersAutoConnectFunction(lastLayer, newLayer)
            lastLayer = newLayer
        return mlp

    @staticmethod
    def RandomFloat() :
        if 'rng' in globals() :
            return rng() / pow(2, 24)
        return random()

    @staticmethod
    def BinaryActivation(sum, gain) :
        x = sum * gain
        return 1.0 if (x >= 0) else 0.0

    @staticmethod
    def SigmoidActivation(sum, gain) :
        x = sum * gain
        return 1.0 / ( 1.0 + exp(-x) )

    @staticmethod
    def TanhActivation(sum, gain) :
        x    = sum * gain
        tanh = 2.0 / (1.0 + exp(-2.0 * x)) - 1.0
        return (tanh / 2.0) + 0.5

    @staticmethod
    def ReLUActivation(sum, gain) :
        x = sum * gain
        return x if x >= 0 else 0.0

    @staticmethod
    def GaussianActivation(sum, gain) :
        x = sum * gain
        return exp(-x ** 2)

    @staticmethod
    def LayersFullConnect(layerSrc, layerDst) :
        if layerSrc and layerDst and layerSrc != layerDst :
            for nSrc in layerSrc.Neurons :
                for nDst in layerDst.Neurons :
                    MicroMLP.Connection(nSrc, nDst)

    @staticmethod
    def GetActivationFunction(name) :
        funcs = {
            MicroMLP.ACTFUNC_BINARY   : MicroMLP.BinaryActivation,
            MicroMLP.ACTFUNC_SIGMOID  : MicroMLP.SigmoidActivation,
            MicroMLP.ACTFUNC_TANH     : MicroMLP.TanhActivation,
            MicroMLP.ACTFUNC_RELU     : MicroMLP.ReLUActivation,
            MicroMLP.ACTFUNC_GAUSSIAN : MicroMLP.GaussianActivation
        }
        return funcs[name] if name in funcs else None
    
    @staticmethod
    def LoadFromFile(filename) :
        try :
            with open(filename, 'r') as jsonFile :
                o = load(jsonFile)
            mlp       = MicroMLP.Create(o['Struct'], o['ActFunc'])
            mlp.Eta   = o['Eta']
            mlp.Alpha = o['Alpha']
            mlp.Gain  = o['Gain']
            for layer in mlp.Layers :
                oLayer = o['Layers'][layer.GetLayerIndex()]
                for neuron in layer.Neurons :
                    oNeuron = oLayer[neuron.GetNeuronIndex()]
                    for oConn in oNeuron :
                        nDst = mlp.GetLayer(oConn['LDst']).GetNeuron(oConn['NDst'])
                        MicroMLP.Connection(neuron, nDst, oConn['Wght'])
            return mlp
        except :
            return None

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
            return self.GetOutputLayer().GetOutputVectorValues()
        return None

    def SaveToFile(self, filename) :
        o = {
            'Eta'     : self.Eta,
            'Alpha'   : self.Alpha,
            'Gain'    : self.Gain,
            'ActFunc' : self._activateFunctionName,
            'Struct'  : [ ],
            'Layers'  : [ ]
        }
        for layer in self.Layers :
            o['Struct'].append(layer.NeuronsCount)
            oLayer = [ ]
            for neuron in layer.Neurons :
                oNeuron = [ ]
                for conn in neuron.GetOutputConnections() :
                    oNeuron.append( {
                        'Wght' : conn.Weight,
                        'LDst' : conn.NeuronDst.ParentLayer.GetLayerIndex(),
                        'NDst' : conn.NeuronDst.GetNeuronIndex()
                    } )
                oLayer.append(oNeuron)
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
                'Output' : targetVectorNNValues
           } )
           return True
        return False

    def ClearExamples(self) :
        self._examples.clear()

    def LearnExamples(self, timeInSec) :
        if self.ExamplesCount > 0 and timeInSec > 0 :
            count   = 0
            endTime = time() + timeInSec
            while time() < endTime :
                idx                  = int( MicroMLP.RandomFloat() * self.ExamplesCount )
                inputVectorNNValues  = self._examples[idx]['Input']
                targetVectorNNValues = self._examples[idx]['Output']
                if not self.Learn(inputVectorNNValues, targetVectorNNValues) :
                    return 0
                count += 1
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
    def ActivateFunctionName(self) :
        return self._activateFunctionName

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
            for layer in self._layers :
                if type(layer) != 'InputLayer' :
                    layer.ComputeLayerValues()
            return True
        return False

    def _backPropagateError(self, training=False) :
        if self.IsNetworkComplete :
            idx = len(self._layers)-2
            while idx >= 0 :
                self._layers[idx].ComputeLayerErrors(training)
                idx -= 1
            return True
        return False

    def _simulate(self, inputVectorNNValues, targetVectorNNValues=None, training=False) :
        if self.IsNetworkComplete and self.GetInputLayer().SetInputVectorNNValues(inputVectorNNValues) :
            self._propagateSignal()
            if not targetVectorNNValues :
                return True
            if self.GetOutputLayer().ComputeTargetLayerError(targetVectorNNValues) :
                return self._backPropagateError(training)
        return False

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

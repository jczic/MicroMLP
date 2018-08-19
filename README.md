## MicroMLP is a micro artificial neural network multilayer perceptron (principally used on ESP32 and [Pycom](http://www.pycom.io) modules)

![HC²](hc2.png "HC²")

####Very easy to integrate and very light with one file only :
- `"microMLP.py"`

####Use deep learning for :
- Signal processing (speech processing, identification, filtering)
- Image processing (compression, recognition, patterns)
- Control (diagnosis, quality control, robotics)
- Optimization (planning, traffic regulation, finance)
- Simulation (black box simulation)
- Classification (DNA analysis)
- Approximation (unknown function, complex function)

<p align="center">
    <img src="nn.png">
</p>

### Using *MicroMLP* static functions :

| Name | Function |
| - | - |
| Create | `mlp = MicroMLP.Create(neuronsByLayers, activateFunctionName, layersAutoConnectFunction=None)` |
| LoadFromFile | `mlp = MicroMLP.LoadFromFile(filename)` |

### Using *MicroMLP* speedly creation of a neural network :
```python
from microMLP import MicroMLP
mlp = MicroMLP.Create([3, 10, 2], "Sigmoid", MicroMLP.LayersFullConnect)
```

### Using *MicroMLP* main class :

| Name | Function |
| - | - |
| Constructor | `mlp = MicroMLP(activateFunctionName)` |
| GetLayer | `layer = mlp.GetLayer(layerIndex)` |
| GetLayerIndex | `idx = mlp.GetLayerIndex(layer)` |
| RemoveLayer | `mlp.RemoveLayer(layer)` |
| GetInputLayer | `inputLayer = mlp.GetInputLayer()` |
| GetOutputLayer | `outputLayer = mlp.GetOutputLayer()` |
| Learn | `ok = mlp.Learn(inputVectorNNValues, targetVectorNNValues)` |
| Test | `ok = mlp.Test(inputVectorNNValues, targetVectorNNValues)` |
| Predict | `outputVectorNNValues = mlp.Predict(inputVectorNNValues)` |
| SaveToFile | `ok = mlp.SaveToFile(filename)` |
| AddExample | `ok = mlp.AddExample(inputVectorNNValues, targetVectorNNValues)` |
| ClearExamples | `mlp.ClearExamples()` |
| LearnExamples | `learnCount = mlp.LearnExamples(maxSeconds=30, maxCount=None, stopWhenLearned=True, printMAEAverage=True)` |

| Property | Example | Read/Write |
| - | - | - |
| Layers | `mlp.Layers` | get |
| LayersCount | `mlp.LayersCount` | get |
| ActivateFunctionName | `mlp.ActivateFunctionName` | get |
| IsNetworkComplete | `mlp.IsNetworkComplete` | get |
| MSE | `mlp.MSE` | get |
| MAE | `mlp.MAE` | get |
| MSEPercent | `mlp.MSEPercent` | get |
| MAEPercent | `mlp.MAEPercent` | get |
| ExamplesCount | `mlp.ExamplesCount` | get |

### Using *MicroMLP* to learn the XOr problem :
```python
from microMLP import MicroMLP

mlp = MicroMLP.Create( neuronsByLayers           = [2, 3, 1],
                       activateFunctionName      = MicroMLP.ACTFUNC_BINARY,
                       layersAutoConnectFunction = MicroMLP.LayersFullConnect )
MicroMLP.Gain = 10

nnFalse = MicroMLP.NNValue.FromBool(False)
nnTrue  = MicroMLP.NNValue.FromBool(True)

mlp.AddExample( [nnFalse, nnFalse], [nnFalse] )
mlp.AddExample( [nnFalse, nnTrue ], [nnTrue ] )
mlp.AddExample( [nnTrue , nnTrue ], [nnFalse] )
mlp.AddExample( [nnTrue , nnFalse], [nnTrue ] )

learnCount = mlp.LearnExamples()

print( "LEARNED :" )
print( "  - False xor False = %s" % mlp.Predict([nnFalse, nnFalse])[0].AsBool )
print( "  - False xor True  = %s" % mlp.Predict([nnFalse, nnTrue] )[0].AsBool )
print( "  - True  xor True  = %s" % mlp.Predict([nnTrue , nnTrue] )[0].AsBool )
print( "  - True  xor False = %s" % mlp.Predict([nnTrue , nnFalse])[0].AsBool )
```

| Variable | Default |
| - | - |
| `mlp.Eta` | 0.30 |
| `mlp.Alpha` | 0.85 |
| `mlp.Gain` | 2.30 |
| `mlp.CorrectLearnedMAE` | 0.02 |

| Graphe | Activation function name | Const | Detail |
| - | - | - | - |
| ![HC²](graphe_binary.png "Binary step") | `"Binary"` | MicroMLP.ACTFUNC_BINARY | Binary step |
| ![HC²](graphe_sigmoid.png "Logistic (sigmoid or soft step)") | `"Sigmoid"` | MicroMLP.ACTFUNC_SIGMOID | Logistic (sigmoid or soft step) |
| ![HC²](graphe_tanh.png "Hyperbolic tangent") | `"TanH"` | MicroMLP.ACTFUNC_TANH | Hyperbolic tangent |
| ![HC²](graphe_relu.png "Rectified linear unit") | `"ReLU"` | MicroMLP.ACTFUNC_RELU | Rectified linear unit |
| ![HC²](graphe_gaussian.png "Gaussian function") | `"Gaussian"` | MicroMLP.ACTFUNC_GAUSSIAN | Gaussian function |

| Layers auto-connect function | Detail |
| - | - |
| `MicroMLP.LayersFullConnect` | Network fully connected |

### Using *MicroMLP.Layer* class :

| Name | Function |
| - | - |
| Constructor | `layer = MicroMLP.Layer(parentMicroMLP, neuronsCount=0, activateFunctionName=None)` |
| GetLayerIndex | `idx = layer.GetLayerIndex()` |
| GetNeuron | `neuron = layer.GetNeuron(neuronIndex)` |
| GetNeuronIndex | `idx = layer.GetNeuronIndex(neuron)` |
| AddNeuron | `layer.AddNeuron(neuron)` |
| RemoveNeuron | `layer.RemoveNeuron(neuron)` |
| GetMeanSquareError | `mse = layer.GetMeanSquareError()` |
| GetMeanAbsoluteError | `mae = layer.GetMeanAbsoluteError()` |
| GetMeanSquareErrorAsPercent | `mseP = layer.GetMeanSquareErrorAsPercent()` |
| GetMeanAbsoluteErrorAsPercent | `maeP = layer.GetMeanAbsoluteErrorAsPercent()` |
| Remove | `layer.Remove()` |

| Property | Example | Read/Write |
| - | - | - |
| ParentMicroMLP | `layer.ParentMicroMLP` | get |
| ActivateFunctionName | `layer.ActivateFunctionName` | get |
| Neurons | `layer.Neurons` | get |
| NeuronsCount | `layer.NeuronsCount` | get |

### Using *MicroMLP.InputLayer(Layer)* class :

| Name | Function |
| - | - |
| Constructor | `inputLayer = MicroMLP.InputLayer(parentMicroMLP, neuronsCount=0)` |
| SetInputVectorNNValues | `ok = inputLayer.SetInputVectorNNValues(inputVectorNNValues)` |

### Using *MicroMLP.OutputLayer(Layer)* class :

| Name | Function |
| - | - |
| Constructor | `outputLayer = MicroMLP.OutputLayer(parentMicroMLP, neuronsCount=0, activateFunctionName=None)` |
| GetOutputVectorNNValues | `outputVectorNNValues = outputLayer.GetOutputVectorNNValues()` |
| ComputeTargetLayerError | `ok = outputLayer.ComputeTargetLayerError(targetVectorNNValues)` |

### Using *MicroMLP.Neuron* class :

| Name | Function |
| - | - |
| Constructor | `neuron = MicroMLP.Neuron(parentLayer, activateFunctionName)` |
| GetNeuronIndex | `idx = neuron.GetNeuronIndex()` |
| GetInputConnections | `connections = neuron.GetInputConnections()` |
| GetOutputConnections | `connections = neuron.GetOutputConnections()` |
| AddInputConnection | `neuron.AddInputConnection(connection)` |
| AddOutputConnection | `neuron.AddOutputConnection(connection)` |
| RemoveInputConnection | `neuron.RemoveInputConnection(connection)` |
| RemoveOutputConnection | `neuron.RemoveOutputConnection(connection)` |
| SetComputedNNValue | `neuron.SetComputedNNValue(nnvalue)` |
| ComputeValue | `neuron.ComputeValue()` |
| ComputeError | `neuron.ComputeError(targetNNValue=None)` |
| Remove | `neuron.Remove()` |

| Property | Example | Read/Write |
| - | - | - |
| ParentLayer | `neuron.ParentLayer` | get |
| ActivateFunctionName | `neuron.ActivateFunctionName` | get |
| ComputedValue | `neuron.ComputedValue` | get |
| ComputedDeltaError | `neuron.ComputedDeltaError` | get |
| ComputedSignalError | `neuron.ComputedSignalError` | get |

### Using *MicroMLP.Connection* class :

| Name | Function |
| - | - |
| Constructor | `connection = MicroMLP.Connection(neuronSrc, neuronDst, weight=None)` |
| UpdateWeight | `connection.UpdateWeight(eta, alpha)` |
| Remove | `connection.Remove()` |

| Property | Example | Read/Write |
| - | - | - |
| NeuronSrc | `connection.NeuronSrc` | get |
| NeuronDst | `connection.NeuronDst` | get |
| Weight | `connection.Weight` | get |

### Using *MicroMLP.NNValue* static functions :

| Name | Function |
| - | - |
| FromPercent | `nnvalue = MicroMLP.NNValue.FromPercent(value)` |
| NewPercent | `nnvalue = MicroMLP.NNValue.NewPercent()` |
| FromByte | `nnvalue = MicroMLP.NNValue.FromByte(value)` |
| NewByte | `nnvalue = MicroMLP.NNValue.NewByte()` |
| FromBool | `nnvalue = MicroMLP.NNValue.FromBool(value)` |
| NewBool | `nnvalue = MicroMLP.NNValue.NewBool()` |
| FromAnalogSignal | `nnvalue = MicroMLP.NNValue.FromAnalogSignal(value)` |
| NewAnalogSignal | `nnvalue = MicroMLP.NNValue.NewAnalogSignal()` |

### Using *MicroMLP.NNValue* class :

| Name | Function |
| - | - |
| Constructor | `nnvalue = MicroMLP.NNValue(minValue, maxValue, value)` |

| Property | Example | Read/Write |
| - | - | - |
| AsFloat | `nnvalue.AsFloat = 639.513` | get / set |
| AsInt | `nnvalue.AsInt = 12345` | get / set |
| AsPercent | `nnvalue.AsPercent = 65` | get / set |
| AsByte | `nnvalue.AsByte = b'\x75'` | get / set |
| AsBool | `nnvalue.AsBool = True` | get / set |
| AsAnalogSignal | `nnvalue.AsAnalogSignal = 0.39472` | get / set |



### By JC`zic for [HC²](https://www.hc2.fr) ;')

*Keep it simple, stupid* :+1:

## MicroMLP is a micro neural network multilayer perceptron (principally used on ESP32 and [Pycom](http://www.pycom.io) modules)

![HC²](hc2.png "HC²")

Very easy to integrate and very light with one file only :
- `"microMLP.py"`

Use deep learning for :
- Signal processing (speech processing, identification, filtering)
- Image processing (compression, recognition, patterns)
- Control (diagnosis, quality control, robotics)
- Optimization (planning, traffic regulation, finance)
- Simulation (black box simulation)
- Classification (DNA analysis)
- Approximation (unknown function, complex function)

### Using *microMLP* static functions :

| Name  | Function |
| - | - |
| Create | `mlp = MicroMLP.Create(neuronsByLayers, activateFunctionName, layersAutoConnectFunction=None)` |
| LoadFromFile | `mlp = MicroMLP.LoadFromFile(filename)` |

### Using *microMLP* main class :

| Name  | Function |
| - | - |
| Constructor | `mlp = MicroMLP(activateFunctionName)` |
| GetLayer | `mlp.GetLayer(layerIndex)` |
| GetLayerIndex | `mlp.GetLayerIndex(layer)` |
| RemoveLayer | `mlp.RemoveLayer(layer)` |
| GetInputLayer | `mlp.GetInputLayer()` |
| GetOutputLayer | `mlp.GetOutputLayer()` |
| Learn | `mlp.Learn(inputVectorNNValues, targetVectorNNValues)` |
| Test | `mlp.Test(inputVectorNNValues, targetVectorNNValues)` |
| Predict | `mlp.GetLayer(inputVectorNNValues)` |
| SaveToFile | `mlp.GetLayer(filename)` |
| AddExample | `mlp.AddExample(inputVectorNNValues, targetVectorNNValues)` |
| ClearExamples | `mlp.ClearExamples()` |
| LearnExamples | `mlp.LearnExamples(timeInSec)` |

| Property  | Example |
| - | - |
| Layers | `mlp.Layers` |
| LayersCount | `mlp.LayersCount` |
| ActivateFunctionName | `mlp.ActivateFunctionName` |
| IsNetworkComplete | `mlp.IsNetworkComplete` |
| MSE | `mlp.MSE` |
| MAE | `mlp.MAE` |
| MSEPercent | `mlp.MSEPercent` |
| MAEPercent | `mlp.MAEPercent` |
| ExamplesCount | `mlp.ExamplesCount` |

| Activation function name | Name |
| 'Binary' | Binary step |
| 'Sigmoid' | Logistic (sigmoid or soft step) |
| 'Tanh' | Hyperbolic tangent |
| 'ReLU' | Rectified linear unit |
| 'Gaussian' | Gaussian function |

| Layers auto-connect function | Detail |
| MicroMLP.LayersFullConnect | Network fully connected |


### By JC`zic for [HC²](https://www.hc2.fr) ;')

*Keep it simple, stupid* :+1:

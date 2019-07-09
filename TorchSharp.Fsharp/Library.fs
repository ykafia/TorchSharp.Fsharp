namespace TorchSharp.FSharp
open TorchSharp
open TorchSharp.Tensor
open TorchSharp.NN
open FsUtils

module FsModule =
    
    let Forward  y (x:ProvidedModule) = x.Forward(y)
    
    let MaxPool2D (kernelSize:list<int64>) (stride:list<int64>) (tensor:TorchTensor)= 
        (kernelSize,stride) 
            |> Tuple2.map List.toArray 
            |> NN.Module.MaxPool2D
            |> Forward tensor
    
    let RelU (inplace:bool) (tensor:TorchTensor) =
        NN.Module.Relu inplace
            |> Forward tensor

    let AdaptiveAvgPool2D (outputSize:list<int64>) (tensor:TorchTensor) =
        outputSize
            |> List.toArray
            |> NN.Module.AdaptiveAvgPool2D
            |> Forward tensor

    let LogSoftMax (dimension:int64) (tensor:TorchTensor)=
        NN.Module.LogSoftMax dimension
            |> Forward tensor

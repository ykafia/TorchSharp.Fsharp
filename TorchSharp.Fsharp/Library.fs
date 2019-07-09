namespace TorchSharp.FSharp
open TorchSharp
open TorchSharp.Tensor
open TorchSharp.NN
module Tuple2 = 
  let map f (a, b) = (f a, f b)
module FsModule =
    
    let Forward  y (x:ProvidedModule) = x.Forward(y)
    
    let MaxPool2D (kernelSize:list<int64>) (stride:list<int64>) (tensor:TorchTensor)= 
        (kernelSize,stride) 
            |> Tuple2.map List.toArray 
            |> NN.Module.MaxPool2D
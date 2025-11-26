CD Examples/TensorStack.Example.Extractors
rmdir /s /q bin\Build\ExtractorDemo
dotnet publish -c Release /p:PublishProfile=SelfContained

CD ..\TensorStack.Example.TextGeneration
rmdir /s /q bin\Build\TextDemo
dotnet publish -c Release /p:PublishProfile=SelfContained

CD ..\TensorStack.Example.Upscaler
rmdir /s /q bin\Build\UpscaleDemo
dotnet publish -c Release /p:PublishProfile=SelfContained

CD ..\..
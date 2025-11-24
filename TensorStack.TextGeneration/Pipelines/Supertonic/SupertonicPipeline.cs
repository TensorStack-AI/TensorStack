using System;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.TextGeneration.Common;

namespace TensorStack.TextGeneration.Pipelines.Supertonic
{
    public class SupertonicPipeline : IPipeline<AudioTensor, SupertonicOptions, GenerateProgress>
    {
        public Task LoadAsync(CancellationToken cancellationToken = default)
        {
            throw new NotImplementedException();
        }


        public Task UnloadAsync(CancellationToken cancellationToken = default)
        {
            throw new NotImplementedException();
        }


        public Task<AudioTensor> RunAsync(SupertonicOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            throw new NotImplementedException();
        }


        public void Dispose()
        {
            throw new NotImplementedException();
        }
    }
}

using Amuse.App.Common;
using Amuse.Common;
using System;
using System.Threading.Tasks;
using TensorStack.Audio;
using TensorStack.Common.Tensor;
using TensorStack.Video;

namespace Amuse.App.Runtime
{
    public interface IDiffusionRuntime : IDisposable
    {
        PipelineModel Pipeline { get; }
        DiffusionDefaultOptions DefaultOptions { get; }
        Task LoadAsync(PipelineModel pipeline, IProgress<PipelineProgress> progressCallback);
        Task ReloadAsync(PipelineModel pipeline, IProgress<PipelineProgress> progressCallback);
        Task UpdateAsync(PipelineModel pipeline);
        Task UnloadAsync();
        Task CancelAsync();
        Task StopAsync();
        Task<ImageTensor> GenerateImageAsync(DiffusionInputOptions options);
        Task<VideoInputStream> GenerateVideoAsync(DiffusionInputOptions options);
        Task<AudioInputStream> GenerateAudioAsync(DiffusionInputOptions options);
        Task<TextResult> GenerateTextAsync(DiffusionInputOptions options);
    }
}

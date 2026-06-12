using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;
using TensorStack.Common;
using TensorStack.Common.Tensor;

namespace Amuse.Common.Message
{
    public sealed class PipelineRequest : IPipelineMessage
    {
        public PipelineRequest() { }
        public PipelineRequest(RequestType type)
        {
            Type = type;
        }

        public PipelineRequest(PipelineLoadOptions options, RequestType type)
        {
            LoadOptions = options;
            Type = type;
        }

        public PipelineRequest(PipelineReloadOptions options)
        {
            ReloadOptions = options;
            Type = RequestType.Reload;
        }

        public PipelineRequest(PipelineCreateOptions options)
        {
            CreateOptions = options;
            Type = RequestType.Create;
        }

        public PipelineRequest(PipelineRunOptions options)
        {
            RunOptions = options;
            Type = RequestType.Run;
            PackTensors();
        }

        public RequestType Type { get; init; }
        public PipelineCreateOptions CreateOptions { get; init; }
        public PipelineLoadOptions LoadOptions { get; init; }
        public PipelineReloadOptions ReloadOptions { get; init; }
        public PipelineRunOptions RunOptions { get; init; }

        public int ImageTensorCount { get; set; }
        public int ControlNetTensorCount { get; set; }
        public int AudioTensorCount { get; set; }

        [JsonIgnore]
        public IReadOnlyList<Tensor<float>> Tensors { get; set; }


        public void PackTensors()
        {
            if (RunOptions == null)
                return;

            ImageTensorCount = RunOptions.InputImages?.Count ?? 0;
            ControlNetTensorCount = RunOptions.InputControlImages?.Count ?? 0;
            AudioTensorCount = RunOptions.InputAudios?.Count ?? 0;
            var totalCount = ImageTensorCount + ControlNetTensorCount + AudioTensorCount;
            if (totalCount > 0)
            {
                var index = 0;
                var validTensors = new Tensor<float>[totalCount];
                if (RunOptions.InputImages != null)
                {
                    foreach (var tensor in RunOptions.InputImages)
                        validTensors[index++] = tensor;
                }

                if (RunOptions.InputControlImages != null)
                {
                    foreach (var tensor in RunOptions.InputControlImages)
                        validTensors[index++] = tensor;
                }

                if (RunOptions.InputAudios != null)
                {
                    foreach (var tensor in RunOptions.InputAudios)
                        validTensors[index++] = tensor;
                }
                Tensors = validTensors;
            }
        }


        public void UnpackTensors()
        {
            if (RunOptions == null || Tensors == null)
                return;

            if (ImageTensorCount > 0)
            {
                RunOptions.InputImages = Tensors
                    .Take(ImageTensorCount)
                    .Select(x => x.AsImageTensor())
                    .ToList();
            }

            if (ControlNetTensorCount > 0)
            {
                RunOptions.InputControlImages = Tensors
                    .Skip(ImageTensorCount)
                    .Take(ControlNetTensorCount)
                    .Select(x => x.AsImageTensor())
                    .ToList();
            }

            if (AudioTensorCount > 0)
            {
                RunOptions.InputAudios = Tensors
                    .Take(AudioTensorCount)
                    .Select(x => x.AsAudioTensor(RunOptions.SampleRate))
                    .ToList();
            }
        }
    }

}

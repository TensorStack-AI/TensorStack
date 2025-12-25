using System.Collections.Generic;
using System.Text.Json.Serialization;
using TensorStack.Common.Tensor;

namespace TensorStack.Python.Common
{
    public record PipelineOptions
    {
        public int Seed { get; set; }
        public string Prompt { get; set; }
        public string NegativePrompt { get; set; }
        public float GuidanceScale { get; set; }
        public float GuidanceScale2 { get; set; }
        public int Steps { get; set; }
        public int Steps2 { get; set; }
        public int Height { get; set; }
        public int Width { get; set; }
        public int Frames { get; set; }
        public float FrameRate { get; set; }
        public float Shift { get; set; }
        public float Strength { get; set; }
        public float ControlNetScale { get; set; }
        public SchedulerType Scheduler { get; set; }
        public List<LoraOptions> LoraOptions { get; set; }

        [JsonIgnore]
        public ImageTensor ImageInput { get; set; }
     
    }
}

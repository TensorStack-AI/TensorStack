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
        public float GuidanceScale { get; set; } = 1;
        public float GuidanceScale2 { get; set; } = 1;
        public int Steps { get; set; } = 50;
        public int Steps2 { get; set; } = 20;
        public int Height { get; set; }
        public int Width { get; set; }
        public int Frames { get; set; }
        public float FrameRate { get; set; }
        public float Shift { get; set; } = 1;
        public float Strength { get; set; } = 1;
        public float ControlNetScale { get; set; } = 1;
        public SchedulerType Scheduler { get; set; }
        public List<LoraOptions> LoraOptions { get; set; }

        [JsonIgnore]
        public ImageTensor InputImage { get; set; }

        [JsonIgnore]
        public ImageTensor InputControlImage { get; set; }
    }
}

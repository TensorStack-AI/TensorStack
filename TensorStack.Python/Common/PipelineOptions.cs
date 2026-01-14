using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;
using TensorStack.Common.Tensor;

namespace TensorStack.Python.Common
{
    public record PipelineOptions
    {
        [JsonPropertyName("seed")]
        public int Seed { get; set; }

        [JsonPropertyName("prompt")]
        public string Prompt { get; set; }

        [JsonPropertyName("negative_prompt")]
        public string NegativePrompt { get; set; }

        [JsonPropertyName("guidance_scale")]
        public float GuidanceScale { get; set; } = 1;

        [JsonPropertyName("guidance_scale2")]
        public float GuidanceScale2 { get; set; } = 1;

        [JsonPropertyName("steps")]
        public int Steps { get; set; } = 50;

        [JsonPropertyName("steps2")]
        public int Steps2 { get; set; } = 20;

        [JsonPropertyName("height")]
        public int Height { get; set; }

        [JsonPropertyName("width")]
        public int Width { get; set; }

        [JsonPropertyName("frames")]
        public int Frames { get; set; }

        [JsonPropertyName("frame_rate")]
        public float FrameRate { get; set; }

        [JsonPropertyName("strength")]
        public float Strength { get; set; } = 1;

        [JsonPropertyName("control_net_scale")]
        public float ControlNetScale { get; set; } = 1;

        [JsonPropertyName("scheduler")]
        public SchedulerType Scheduler { get; set; }

        [JsonPropertyName("scheduler_options")]
        public SchedulerOptions SchedulerOptions { get; set; }

        [JsonPropertyName("lora_options")]
        public List<LoraOptions> LoraOptions { get; set; }


        [JsonIgnore]
        public ImageTensor InputImage
        {
            get { return InputImages.FirstOrDefault(); }
            set
            {
                if (InputImages.Count == 0)
                {
                    InputImages.Add(value);
                }
                else
                {
                    InputImages[0] = value;
                }
            }
        }

        [JsonIgnore]
        public ImageTensor InputControlImage
        {
            get { return InputControlImages.FirstOrDefault(); }
            set
            {
                if (InputControlImages.Count == 0)
                {
                    InputControlImages.Add(value);
                }
                else
                {
                    InputControlImages[0] = value;
                }
            }
        }


        [JsonIgnore]
        public List<ImageTensor> InputImages { get; set; } = [];

        [JsonIgnore]
        public List<ImageTensor> InputControlImages { get; set; } = [];

    }
}

using System.Collections.Generic;
using System.Text.Json.Serialization;
using TensorStack.Common.Tensor;

namespace Amuse.Common
{
    public sealed record PipelineRunOptions
    {
        public int Seed { get; set; }
        public string Prompt { get; set; }
        public string Prompt2 { get; set; }
        public string NegativePrompt { get; set; }
        public float GuidanceScale { get; set; } = 1;
        public float GuidanceScale2 { get; set; } = 1;
        public int Steps { get; set; } = 50;
        public int Steps2 { get; set; } = 20;
        public int Height { get; set; }
        public int Width { get; set; }
        public int Frames { get; set; }
        public float FrameRate { get; set; }
        public float Strength { get; set; } = 1;
        public float ControlNetScale { get; set; } = 1;
        public string TempFileName { get; set; }
        public int FrameChunk { get; set; }
        public int FrameChunkOverlap { get; set; }
        public int NoiseCondition { get; set; }
        public bool EnableVaeTiling { get; set; }
        public bool EnableVaeSlicing { get; set; }
        public float Duration { get; set; } = 5f;
        public string Instruction { get; set; }
        public int MaxLength { get; set; }
        public int MaxLength2 { get; set; }
        public int Bpm { get; set; }
        public string Keyscale { get; set; }
        public string TimeSignature { get; set; }
        public string Task { get; set; }
        public string TrackName { get; set; }
        public float Speed { get; set; }
        public float SilenceDuration { get; set; }
        public int MinLength { get; set; } = 20;
        public int NoRepeatNgramSize { get; set; } = 3;
        public int Beams { get; set; } = 1;
        public int TopK { get; set; } = 1;
        public float TopP { get; set; } = 0.9f;
        public float Temperature { get; set; } = 1.0f;
        public float LengthPenalty { get; set; } = 1.0f;
        public string EarlyStopping { get; set; }
        public int DiversityLength { get; set; } = 20;
        public bool OutputLastHiddenStates { get; set; }
        public int ChunkSize { get; set; }
        public LanguageType Language { get; set; }
        public int SampleRate { get; set; }

        public SchedulerOptions SchedulerOptions { get; set; }
        public List<LoraOptions> LoraOptions { get; set; }


        [JsonIgnore]
        public List<ImageTensor> InputImages { get; set; } = [];

        [JsonIgnore]
        public List<ImageTensor> InputControlImages { get; set; } = [];

        [JsonIgnore]
        public List<AudioTensor> InputAudios { get; set; } = [];
    }



}

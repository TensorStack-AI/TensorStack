using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record FlowMatchLCMOptions : SchedulerOptions
    {
        public FlowMatchLCMOptions() : base() { }
        private FlowMatchLCMOptions(FlowMatchLCMOptions other) : base(other)
        {
            ShallowCopyProperties(other);
            ScaleFactors = other.ScaleFactors?.ToList();
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.FlowMatchLCM;

        [JsonPropertyName("num_train_timesteps")]
        public int NumTrainTimesteps { get; init; } = 1000;

        [JsonPropertyName("shift")]
        public float Shift { get; init; } = 1.0f;

        [JsonPropertyName("use_dynamic_shifting")]
        public bool UseDynamicShifting { get; init; } = false;

        [JsonPropertyName("base_shift")]
        public float? BaseShift { get; init; } = 0.5f;

        [JsonPropertyName("max_shift")]
        public float? MaxShift { get; init; } = 1.15f;

        [JsonPropertyName("base_image_seq_len")]
        public int BaseImageSeqLen { get; set; } = 256;

        [JsonPropertyName("max_image_seq_len")]
        public int MaxImageSeqLen { get; set; } = 4096;

        [JsonPropertyName("invert_sigmas")]
        public bool InvertSigmas { get; set; }

        [JsonPropertyName("shift_terminal")]
        public float? ShiftTerminal { get; set; } = null;

        [JsonPropertyName("use_karras_sigmas")]
        public bool UseKarrasSigmas { get; set; }

        [JsonPropertyName("use_exponential_sigmas")]
        public bool UseExponentialSigmas { get; set; }

        [JsonPropertyName("use_beta_sigmas")]
        public bool UseBetaSigmas { get; set; }

        [JsonPropertyName("time_shift_type")]
        public TimeShiftType TimeShiftType { get; set; } = TimeShiftType.Exponential;

        [JsonPropertyName("scale_factors")]
        public List<float> ScaleFactors { get; init; }

        [JsonPropertyName("upscale_mode")]
        public UpscaleModeType UpscaleMode { get; set; } = UpscaleModeType.Bicubic;
    }
}

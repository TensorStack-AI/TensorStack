using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record HeliosDMDOptions : SchedulerOptions
    {
        public HeliosDMDOptions() : base() { }
        private HeliosDMDOptions(HeliosDMDOptions other) : base(other)
        {
            ShallowCopyProperties(other);
            StageRange = other.StageRange?.ToList();
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.HeliosDMD;

        [JsonPropertyName("num_train_timesteps")]
        public int NumTrainTimesteps { get; init; } = 1000;

        [JsonPropertyName("shift")]
        public float Shift { get; init; } = 1f;

        [JsonPropertyName("stages")]
        public int Stages { get; init; } = 3;

        [JsonPropertyName("stage_range")]
        public List<float> StageRange { get; init; } = [0f, 1.0f / 3.0f, 2.0f / 3.0f, 1f];

        [JsonPropertyName("gamma")]
        public float Gamma { get; set; } = 1f / 3f;

        [JsonPropertyName("prediction_type")]
        public PredictionType PredictionType { get; init; } = PredictionType.FlowPrediction;

        [JsonPropertyName("use_flow_sigmas")]
        public bool UseFlowSigmas { get; set; } = true;

        [JsonPropertyName("use_dynamic_shifting")]
        public bool UseDynamicShifting { get; init; } = false;

        [JsonPropertyName("time_shift_type")]
        public TimeShiftType TimeShiftType { get; set; } = TimeShiftType.Linear;
    }
}

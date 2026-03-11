using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record EulerAncestralOptions : SchedulerOptions
    {
        public EulerAncestralOptions() : base() { }
        private EulerAncestralOptions(EulerAncestralOptions other) : base(other)
        {
            ShallowCopyProperties(other);
            TrainedBetas = other.TrainedBetas?.ToList();
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.EulerAncestral;

        [JsonPropertyName("num_train_timesteps")]
        public int NumTrainTimesteps { get; init; } = 1000;

        [JsonPropertyName("beta_start")]
        public float BetaStart { get; init; } = 0.00085f;

        [JsonPropertyName("beta_end")]
        public float BetaEnd { get; init; } = 0.012f;

        [JsonPropertyName("beta_schedule")]
        public BetaScheduleType BetaSchedule { get; init; } = BetaScheduleType.ScaledLinear;

        [JsonPropertyName("trained_betas")]
        public List<float> TrainedBetas { get; set; }

        [JsonPropertyName("prediction_type")]
        public PredictionType PredictionType { get; init; } = PredictionType.Epsilon;

        [JsonPropertyName("timestep_spacing")]
        public TimestepSpacingType TimestepSpacing { get; init; } = TimestepSpacingType.Linspace;

        [JsonPropertyName("steps_offset")]
        public int StepsOffset { get; init; } = 0;

        [JsonPropertyName("rescale_betas_zero_snr")]
        public bool RescaleBetasZeroSNR { get; set; }
    }
}

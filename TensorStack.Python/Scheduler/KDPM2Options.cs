using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record KDPM2Options : SchedulerOptions
    {
        public KDPM2Options() : base() { }
        private KDPM2Options(KDPM2Options other) : base(other)
        {
            ShallowCopyProperties(other);
            TrainedBetas = other.TrainedBetas?.ToList();
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.KDPM2;

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

        [JsonPropertyName("use_karras_sigmas")]
        public bool UseKarrasSigmas { get; set; }

        [JsonPropertyName("use_exponential_sigmas")]
        public bool UseExponentialSigmas { get; set; }

        [JsonPropertyName("use_beta_sigmas")]
        public bool UseBetaSigmas { get; set; }

        [JsonPropertyName("prediction_type")]
        public PredictionType PredictionType { get; init; } = PredictionType.Epsilon;

        [JsonPropertyName("timestep_spacing")]
        public TimestepSpacingType TimestepSpacing { get; init; } = TimestepSpacingType.Trailing;

        [JsonPropertyName("steps_offset")]
        public int StepsOffset { get; init; } = 0;
    }
}

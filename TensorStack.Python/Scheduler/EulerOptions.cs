using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record EulerOptions : SchedulerOptions
    {
        public EulerOptions() : base() { }
        private EulerOptions(EulerOptions other) : base(other)
        {
            ShallowCopyProperties(other);
            TrainedBetas = other.TrainedBetas?.ToList();
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.Euler;

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

        [JsonPropertyName("interpolation_type")]
        public InterpolationType InterpolationType { get; set; } = InterpolationType.Linear;

        [JsonPropertyName("use_karras_sigmas")]
        public bool UseKarrasSigmas { get; set; }

        [JsonPropertyName("use_exponential_sigmas")]
        public bool UseExponentialSigmas { get; set; }

        [JsonPropertyName("use_beta_sigmas")]
        public bool UseBetaSigmas { get; set; }

        [JsonPropertyName("prediction_type")]
        public PredictionType PredictionType { get; init; } = PredictionType.Epsilon;

        [JsonPropertyName("sigma_min")]
        public float? SigmaMin { get; set; } = null;

        [JsonPropertyName("sigma_max")]
        public float? SigmaMax { get; set; } = null;

        [JsonPropertyName("timestep_spacing")]
        public TimestepSpacingType TimestepSpacing { get; init; } = TimestepSpacingType.Linspace;

        [JsonPropertyName("timestep_type")]
        public TimestepType TimestepType { get; init; } = TimestepType.Discrete;

        [JsonPropertyName("steps_offset")]
        public int StepsOffset { get; init; } = 0;

        [JsonPropertyName("rescale_betas_zero_snr")]
        public bool RescaleBetasZeroSNR { get; set; }

        [JsonPropertyName("final_sigmas_type")]
        public FinalSigmasType FinalSigmasType { get; set; } = FinalSigmasType.Zero;
    }
}

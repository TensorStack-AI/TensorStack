using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record CogVideoXDDIMOptions : SchedulerOptions
    {
        public CogVideoXDDIMOptions() : base() { }
        private CogVideoXDDIMOptions(CogVideoXDDIMOptions other) : base(other)
        {
            ShallowCopyProperties(other);
            TrainedBetas = other.TrainedBetas?.ToList();
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.CogVideoXDDIM;

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

        [JsonPropertyName("set_alpha_to_one")]
        public bool SetAlphaToOne { get; set; } = true;

        [JsonPropertyName("prediction_type")]
        public PredictionType PredictionType { get; init; } = PredictionType.Epsilon;

        [JsonPropertyName("clip_sample")]
        public bool ClipSample { get; set; } = true;

        [JsonPropertyName("clip_sample_range")]
        public float ClipSampleRange { get; set; } = 1f;

        [JsonPropertyName("sample_max_value")]
        public float SampleMaxValue { get; set; } = 1f;

        [JsonPropertyName("timestep_spacing")]
        public TimestepSpacingType TimestepSpacing { get; init; } = TimestepSpacingType.Leading;

        [JsonPropertyName("steps_offset")]
        public int StepsOffset { get; init; } = 0;

        [JsonPropertyName("rescale_betas_zero_snr")]
        public bool RescaleBetasZeroSNR { get; set; }

        [JsonPropertyName("snr_shift_scale")]
        public float SNRShiftScale { get; set; } = 3f;
    }
}

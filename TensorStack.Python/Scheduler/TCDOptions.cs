using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record TCDOptions : SchedulerOptions
    {
        public TCDOptions() : base() { }
        private TCDOptions(TCDOptions other) : base(other)
        {
            ShallowCopyProperties(other);
            TrainedBetas = other.TrainedBetas?.ToList();
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.TCD;

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

        [JsonPropertyName("original_inference_steps")]
        public int OriginalInferenceSteps { get; set; } = 50;

        [JsonPropertyName("clip_sample")]
        public bool ClipSample { get; set; } = true;

        [JsonPropertyName("clip_sample_range")]
        public float ClipSampleRange { get; set; } = 1f;

        [JsonPropertyName("set_alpha_to_one")]
        public bool SetAlphaToOne { get; set; } = true;

        [JsonPropertyName("steps_offset")]
        public int StepsOffset { get; init; } = 0;

        [JsonPropertyName("prediction_type")]
        public PredictionType PredictionType { get; init; } = PredictionType.Epsilon;

        [JsonPropertyName("thresholding")]
        public bool Thresholding { get; set; }

        [JsonPropertyName("dynamic_thresholding_ratio")]
        public float DynamicThresholdingRatio { get; set; } = 0.995f;

        [JsonPropertyName("sample_max_value")]
        public float SampleMaxValue { get; set; } = 1f;

        [JsonPropertyName("timestep_spacing")]
        public TimestepSpacingType TimestepSpacing { get; init; } = TimestepSpacingType.Leading;

        [JsonPropertyName("timestep_scaling")]
        public float TimestepScaling { get; set; } = 10.0f;

        [JsonPropertyName("rescale_betas_zero_snr")]
        public bool RescaleBetasZeroSNR { get; set; }
    }
}

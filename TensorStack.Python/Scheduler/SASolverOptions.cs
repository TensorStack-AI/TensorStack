using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record SASolverOptions : SchedulerOptions
    {
        public SASolverOptions() : base() { }
        private SASolverOptions(SASolverOptions other) : base(other)
        {
            ShallowCopyProperties(other);
            TrainedBetas = other.TrainedBetas?.ToList();
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.SASolver;

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

        [JsonPropertyName("predictor_order")]
        public int PredictorOrder { get; init; } = 2;

        [JsonPropertyName("corrector_order")]
        public int CorrectorOrder { get; init; } = 2;

        [JsonPropertyName("prediction_type")]
        public PredictionType PredictionType { get; init; } = PredictionType.Epsilon;

        [JsonPropertyName("thresholding")]
        public bool Thresholding { get; set; }

        [JsonPropertyName("dynamic_thresholding_ratio")]
        public float DynamicThresholdingRatio { get; set; } = 0.995f;

        [JsonPropertyName("sample_max_value")]
        public float SampleMaxValue { get; set; } = 1f;

        [JsonPropertyName("algorithm_type")]
        public AlgorithmType AlgorithmType { get; set; } = AlgorithmType.DataPrediction;

        [JsonPropertyName("lower_order_final")]
        public bool LowerOrderFinal { get; set; } = true;

        [JsonPropertyName("use_karras_sigmas")]
        public bool UseKarrasSigmas { get; set; }

        [JsonPropertyName("use_exponential_sigmas")]
        public bool UseExponentialSigmas { get; set; }

        [JsonPropertyName("use_beta_sigmas")]
        public bool UseBetaSigmas { get; set; }

        [JsonPropertyName("use_flow_sigmas")]
        public bool UseFlowSigmas { get; set; }

        [JsonPropertyName("flow_shift")]
        public float FlowShift { get; set; } = 1f;

        [JsonPropertyName("variance_type")]
        public VarianceType? VarianceType { get; set; } = null;

        [JsonPropertyName("timestep_spacing")]
        public TimestepSpacingType TimestepSpacing { get; init; } = TimestepSpacingType.Linspace;

        [JsonPropertyName("steps_offset")]
        public int StepsOffset { get; init; } = 0;
    }
}

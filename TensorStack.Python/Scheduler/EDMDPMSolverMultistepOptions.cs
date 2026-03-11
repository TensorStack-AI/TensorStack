using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record EDMDPMSolverMultistepOptions : SchedulerOptions
    {
        public EDMDPMSolverMultistepOptions() : base() { }
        private EDMDPMSolverMultistepOptions(EDMDPMSolverMultistepOptions other) : base(other)
        {
            ShallowCopyProperties(other);
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.EDMDPMSolverMultistep;

        [JsonPropertyName("num_train_timesteps")]
        public int NumTrainTimesteps { get; init; } = 1000;

        [JsonPropertyName("sigma_min")]
        public float SigmaMin { get; init; } = 0.002f;

        [JsonPropertyName("sigma_max")]
        public float SigmaMax { get; init; } = 80.0f;

        [JsonPropertyName("sigma_data")]
        public float SigmaData { get; init; } = 0.5f;

        [JsonPropertyName("sigma_schedule")]
        public SigmaScheduleType SigmaScheduleType { get; init; } = SigmaScheduleType.Karras;

        [JsonPropertyName("rho")]
        public float Rho { get; init; } = 7.0f;

        [JsonPropertyName("solver_order")]
        public int SolverOrder { get; set; } = 2;

        [JsonPropertyName("prediction_type")]
        public PredictionType PredictionType { get; init; } = PredictionType.Epsilon;

        [JsonPropertyName("thresholding")]
        public bool Thresholding { get; set; }

        [JsonPropertyName("dynamic_thresholding_ratio")]
        public float DynamicThresholdingRatio { get; set; } = 0.995f;

        [JsonPropertyName("sample_max_value")]
        public float SampleMaxValue { get; set; } = 1f;

        [JsonPropertyName("algorithm_type")]
        public AlgorithmType AlgorithmType { get; set; } = AlgorithmType.DPMSolverPlus;

        [JsonPropertyName("solver_type")]
        public SolverType SolverType { get; set; } = SolverType.LogRho;

        [JsonPropertyName("lower_order_final")]
        public bool LowerOrderFinal { get; set; } = true;

        [JsonPropertyName("euler_at_final")]
        public bool EulerAtFinal { get; set; } = false;

        [JsonPropertyName("final_sigmas_type")]
        public FinalSigmasType FinalSigmasType { get; init; } = FinalSigmasType.Zero;
    }
}

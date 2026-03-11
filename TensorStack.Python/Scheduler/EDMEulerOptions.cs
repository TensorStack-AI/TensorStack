using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record EDMEulerOptions : SchedulerOptions
    {
        public EDMEulerOptions() : base() { }
        private EDMEulerOptions(EDMEulerOptions other) : base(other)
        {
            ShallowCopyProperties(other);
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.EDMEuler;

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

        [JsonPropertyName("prediction_type")]
        public PredictionType PredictionType { get; init; } = PredictionType.Epsilon;

        [JsonPropertyName("rho")]
        public float Rho { get; init; } = 7.0f;

        [JsonPropertyName("final_sigmas_type")]
        public FinalSigmasType FinalSigmasType { get; init; } = FinalSigmasType.Zero;
    }
}

using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record SCMOptions : SchedulerOptions
    {
        public SCMOptions() : base() { }
        private SCMOptions(SCMOptions other) : base(other)
        {
            ShallowCopyProperties(other);
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.SCM;

        [JsonPropertyName("num_train_timesteps")]
        public int NumTrainTimesteps { get; init; } = 1000;

        [JsonPropertyName("prediction_type")]
        public PredictionType PredictionType { get; init; } = PredictionType.Trigflow;

        [JsonPropertyName("sigma_data")]
        public float SigmaData { get; set; } = 0.5f;
    }
}

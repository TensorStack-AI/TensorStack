using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record LTXEulerAncestralRFOptions : SchedulerOptions
    {
        public LTXEulerAncestralRFOptions() : base() { }
        private LTXEulerAncestralRFOptions(LTXEulerAncestralRFOptions other) : base(other)
        {
            ShallowCopyProperties(other);
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.LTXEulerAncestral;

        [JsonPropertyName("num_train_timesteps")]
        public int NumTrainTimesteps { get; init; } = 1000;

        [JsonPropertyName("eta")]
        public float Eta { get; set; } = 1f;

        [JsonPropertyName("s_noise")]
        public float SNoise { get; set; } = 1f;
    }
}

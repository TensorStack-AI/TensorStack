using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record DDPMWuerstchenOptions : SchedulerOptions
    {
        public DDPMWuerstchenOptions() : base() { }
        private DDPMWuerstchenOptions(DDPMWuerstchenOptions other) : base(other)
        {
            ShallowCopyProperties(other);
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.DDPMWuerstchen;

        [JsonPropertyName("scaler")]
        public float Scaler { get; init; } = 1f;

        [JsonPropertyName("s")]
        public float S { get; init; } = 0.008f;
    }
}

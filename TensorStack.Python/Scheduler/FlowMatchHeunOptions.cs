using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record FlowMatchHeunOptions : SchedulerOptions
    {
        public FlowMatchHeunOptions() : base() { }
        private FlowMatchHeunOptions(FlowMatchHeunOptions other) : base(other)
        {
            ShallowCopyProperties(other);
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.FlowMatchHeun;

        [JsonPropertyName("num_train_timesteps")]
        public int NumTrainTimesteps { get; init; } = 1000;

        [JsonPropertyName("shift")]
        public float Shift { get; init; } = 1.0f;
    }
}

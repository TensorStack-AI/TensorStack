using System.Collections.Generic;
using System.Linq;
using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public sealed record HeliosOptions : SchedulerOptions
    {
        public HeliosOptions() : base() { }
        private HeliosOptions(HeliosOptions other) : base(other)
        {
            ShallowCopyProperties(other);
            StageRange = other.StageRange?.ToList();
            DisableCorrector = other.DisableCorrector?.ToList();
        }

        [JsonIgnore]
        public override SchedulerType Scheduler => SchedulerType.Helios;

        [JsonPropertyName("num_train_timesteps")]
        public int NumTrainTimesteps { get; init; } = 1000;

        [JsonPropertyName("shift")]
        public float Shift { get; init; } = 1f;

        [JsonPropertyName("stages")]
        public int Stages { get; init; } = 3;

        [JsonPropertyName("stage_range")]
        public List<float> StageRange { get; init; } = [0f, 1.0f / 3.0f, 2.0f / 3.0f, 1f];

        [JsonPropertyName("gamma")]
        public float Gamma { get; set; } = 1f / 3f;

        [JsonPropertyName("thresholding")]
        public bool Thresholding { get; set; }

        [JsonPropertyName("prediction_type")]
        public PredictionType PredictionType { get; init; } = PredictionType.FlowPrediction;

        [JsonPropertyName("solver_order")]
        public int SolverOrder { get; init; } = 2;

        [JsonPropertyName("predict_x0")]
        public bool PredictX0 { get; set; } = true;

        [JsonPropertyName("solver_type")]
        public SolverType SolverType { get; set; } = SolverType.BH2;

        [JsonPropertyName("lower_order_final")]
        public bool LowerOrderFinal { get; set; } = true;

        [JsonPropertyName("disable_corrector")]
        public List<int> DisableCorrector { get; init; } = [];

        [JsonPropertyName("use_flow_sigmas")]
        public bool UseFlowSigmas { get; set; } = true;

        [JsonPropertyName("use_dynamic_shifting")]
        public bool UseDynamicShifting { get; init; } = false;

        [JsonPropertyName("time_shift_type")]
        public TimeShiftType TimeShiftType { get; set; } = TimeShiftType.Exponential;
    }
}

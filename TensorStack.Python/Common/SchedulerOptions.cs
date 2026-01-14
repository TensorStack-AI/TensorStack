using System.Text.Json.Serialization;

namespace TensorStack.Python.Common
{
    public sealed record SchedulerOptions
    {
        public int NumTrainTimesteps { get; init; } = 1000;
        public int StepsOffset { get; init; } = 0;

        // IsTimestep
        public float BetaStart { get; init; } = 0.00085f;
        public float BetaEnd { get; init; } = 0.012f;
        public BetaScheduleType BetaSchedule { get; init; } = BetaScheduleType.ScaledLinear;
        public PredictionType PredictionType { get; init; } = PredictionType.Epsilon;
        public VarianceType? VarianceType { get; init; }
        public TimestepSpacingType TimestepSpacing { get; init; } = TimestepSpacingType.Linspace;

        // IsClipSample
        public bool ClipSample { get; init; } = false;
        public float ClipSampleRange { get; init; } = 1.0f;

        //IsThreshold
        public bool Thresholding { get; init; } = false;
        public float DynamicThresholdingRatio { get; init; } = 0.995f;
        public float SampleMaxValue { get; init; } = 1.0f;

        // IsKarras
        public bool UseKarrasSigmas { get; init; } = false;
        public float? SigmaMin { get; init; }
        public float? SigmaMax { get; init; }
        public float Rho { get; init; } = 7.0f;

        // IsMultiStep
        public int SolverOrder { get; init; } = 2; //  Usually 1–3
        public SolverType SolverType { get; init; } = SolverType.Midpoint;
        public AlgorithmType AlgorithmType { get; init; } = AlgorithmType.DPMSolverPlus;
        public bool LowerOrderFinal { get; init; } = true;

        // IsStochastic
        public float Eta { get; init; } = 0.0f;
        public float SNoise { get; init; } = 1.0f;
        public float SChurn { get; init; } = 0.0f;
        public float STmin { get; init; } = 0.0f;
        public float STmax { get; init; } = 0.0f; // 0 = float.PositiveInfinity;


        // IsFlowMatch
        public float Shift { get; init; } = 1.0f;
        public bool UseDynamicShifting { get; init; } = false;
        public float BaseShift { get; init; } = 0.5f;
        public float MaxShift { get; init; } = 1.15f;
        public bool StochasticSampling { get; init; } = false;
        public float FlowShift => Shift;
    }


    public enum TimestepSpacingType
    {
        [JsonStringEnumMemberName("leading")]
        Leading = 0,

        [JsonStringEnumMemberName("trailing")]
        Trailing = 1,

        [JsonStringEnumMemberName("linspace")]
        Linspace = 2
    }


    public enum AlgorithmType
    {
        [JsonStringEnumMemberName("dpmsolver")]
        DPMSolver = 0,

        [JsonStringEnumMemberName("dpmsolver++")]
        DPMSolverPlus = 1
    }


    public enum SolverType
    {

        [JsonStringEnumMemberName("midpoint")]
        Midpoint = 0,

        [JsonStringEnumMemberName("heun")]
        Heun = 1
    }


    public enum BetaScheduleType
    {
        [JsonStringEnumMemberName("linear")]
        Linear = 0,

        [JsonStringEnumMemberName("scaled_linear")]
        ScaledLinear = 1,

        [JsonStringEnumMemberName("cosine")]
        Cosine = 2
    }


    public enum PredictionType
    {
        [JsonStringEnumMemberName("epsilon")]
        Epsilon = 0,

        [JsonStringEnumMemberName("v_prediction")]
        Variable = 1,

        [JsonStringEnumMemberName("sample")]
        Sample = 2
    }


    public enum VarianceType
    {
        [JsonStringEnumMemberName("fixed_small")]
        FixedSmall = 0,

        [JsonStringEnumMemberName("fixed_large")]
        FixedLarge = 1,

        [JsonStringEnumMemberName("learned")]
        Learned = 2,

        [JsonStringEnumMemberName("learned_range")]
        LearnedRange = 3
    }
}

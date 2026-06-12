using System.Collections.Generic;

namespace Amuse.Common
{
    public sealed record SchedulerOptions
    {
        public SchedulerType Scheduler { get; set; }
        public int NumTrainTimesteps { get; set; }
        public int OriginalInferenceSteps { get; set; }
        public int BaseImageSeqLen { get; set; }
        public int MaxImageSeqLen { get; set; }
        public BetaScheduleType BetaSchedule { get; set; }
        public float BetaStart { get; set; }
        public float BetaEnd { get; set; }
        public PredictionType PredictionType { get; set; }
        public TimestepSpacingType TimestepSpacing { get; set; }
        public int StepsOffset { get; set; }
        public bool ClipSample { get; set; }
        public float ClipSampleRange { get; set; }
        public float SampleMaxValue { get; set; }
        public bool Thresholding { get; set; }
        public float DynamicThresholdingRatio { get; set; }
        public VarianceType? VarianceType { get; set; }
        public bool UseKarrasSigmas { get; set; }
        public bool UseBetaSigmas { get; set; }
        public bool UseExponentialSigmas { get; set; }
        public bool UseFlowSigmas { get; set; }
        public float? SigmaMin { get; set; }
        public float? SigmaMax { get; set; }
        public FinalSigmasType FinalSigmasType { get; set; }
        public InterpolationType InterpolationType { get; set; }
        public TimestepType TimestepType { get; set; }
        public bool RescaleBetasZeroSNR { get; set; }
        public bool SetAlphaToOne { get; set; }
        public float TimestepScaling { get; set; }
        public float Shift { get; set; }
        public float BaseShift { get; set; }
        public float MaxShift { get; set; }
        public float? ShiftTerminal { get; set; }
        public bool UseDynamicShifting { get; set; }
        public float FlowShift { get; set; }
        public float SNRShiftScale { get; set; }
        public TimeShiftType TimeShiftType { get; set; }
        public float Rho { get; set; }
        public int SolverOrder { get; set; }
        public SolverType SolverType { get; set; }
        public AlgorithmType AlgorithmType { get; set; }
        public bool LowerOrderFinal { get; set; }
        public bool StochasticSampling { get; set; }
        public float Eta { get; set; }
        public float SNoise { get; set; }
        public bool InvertSigmas { get; set; }
        public bool SkipPrkSteps { get; set; }
        public bool PredictX0 { get; set; }
        public bool EulerAtFinal { get; set; }
        public bool UseLuLambdas { get; set; }
        public int? NoiseSamplerSeed { get; set; }
        public float SigmaData { get; set; }
        public SigmaScheduleType SigmaScheduleType { get; set; }
        public UpscaleModeType UpscaleMode { get; set; }
        public int Stages { get; set; }
        public float Gamma { get; set; }
        public int PredictorOrder { get; set; }
        public int CorrectorOrder { get; set; }
        public List<float> ScaleFactors { get; set; }
        public List<float> StageRange { get; set; }
        public List<int> DisableCorrector { get; set; }
        public float SValue { get; set; }
        public float Scaler { get; set; }
    }



}

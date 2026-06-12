using Amuse.Common;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;
using TensorStack.WPF;

namespace Amuse.App.Common
{
    public sealed record SchedulerInputOptions : BaseRecord
    {
        private SchedulerType _scheduler;
        private int _numTrainTimesteps = 0;
        private int _originalInferenceSteps;
        private int _baseImageSeqLen;
        private int _maxImageSeqLen;
        private BetaScheduleType _betaSchedule;
        private float _betaStart = 0f;
        private float _betaEnd = 0f;
        private PredictionType _predictionType;
        private TimestepSpacingType _timestepSpacing;
        private int _stepsOffset = 0;
        private bool _clipSample = false;
        private float _clipSampleRange = 0f;
        private float _sampleMaxValue = 0f;
        private bool _thresholding = false;
        private float _dynamicThresholdingRatio = 0f;
        private VarianceType? _varianceType;
        private bool _useKarrasSigmas;
        private bool _useBetaSigmas;
        private bool _useExponentialSigmas;
        private bool _useFlowSigmas;
        private float _sigmaMin;
        private float _sigmaMax;
        private FinalSigmasType _finalSigmasType;
        private InterpolationType _interpolationType;
        private TimestepType _timestepType;
        private bool _rescaleBetasZeroSNR;
        private bool _setAlphaToOne;
        private float _timestepScaling;
        private float _shift = 0f;
        private float _baseShift = 0f;
        private float _maxShift = 0f;
        private float _shiftTerminal;
        private bool _useDynamicShifting;
        private float _flowShift = 0;
        private float _sNRShiftScale;
        private TimeShiftType _timeShiftType;
        private float _rho = 0f;
        private int _solverOrder = 0;
        private SolverType _solverType;
        private AlgorithmType _algorithmType;
        private bool _lowerOrderFinal;
        private bool _stochasticSampling;
        private float _eta = 0.0f;
        private float _sNoise = 0f;
        private bool _invertSigmas;
        private bool _skipPrkSteps;
        private bool _predictX0;
        private bool _eulerAtFinal;
        private bool _useLuLambdas;
        private int _noiseSamplerSeed;
        private float _sigmaData;
        private SigmaScheduleType _sigmaScheduleType;
        private UpscaleModeType _upscaleMode;
        private int _stages;
        private float _gamma;
        private int _predictorOrder;
        private int _correctorOrder;
        private List<float> _scaleFactors;
        private List<float> _stageRange;
        private List<int> _disableCorrector;
        private float _sValue;
        private float _scaler;

        public SchedulerType Scheduler
        {
            get { return _scheduler; }
            set { SetProperty(ref _scheduler, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public int NumTrainTimesteps
        {
            get { return _numTrainTimesteps; }
            set { SetProperty(ref _numTrainTimesteps, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public int OriginalInferenceSteps
        {
            get { return _originalInferenceSteps; }
            set { SetProperty(ref _originalInferenceSteps, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public int BaseImageSeqLen
        {
            get { return _baseImageSeqLen; }
            set { SetProperty(ref _baseImageSeqLen, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public int MaxImageSeqLen
        {
            get { return _maxImageSeqLen; }
            set { SetProperty(ref _maxImageSeqLen, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public BetaScheduleType BetaSchedule
        {
            get { return _betaSchedule; }
            set { SetProperty(ref _betaSchedule, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float BetaStart
        {
            get { return _betaStart; }
            set { SetProperty(ref _betaStart, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float BetaEnd
        {
            get { return _betaEnd; }
            set { SetProperty(ref _betaEnd, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public PredictionType PredictionType
        {
            get { return _predictionType; }
            set { SetProperty(ref _predictionType, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public TimestepSpacingType TimestepSpacing
        {
            get { return _timestepSpacing; }
            set { SetProperty(ref _timestepSpacing, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public int StepsOffset
        {
            get { return _stepsOffset; }
            set { SetProperty(ref _stepsOffset, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool ClipSample
        {
            get { return _clipSample; }
            set { SetProperty(ref _clipSample, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float ClipSampleRange
        {
            get { return _clipSampleRange; }
            set { SetProperty(ref _clipSampleRange, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float SampleMaxValue
        {
            get { return _sampleMaxValue; }
            set { SetProperty(ref _sampleMaxValue, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool Thresholding
        {
            get { return _thresholding; }
            set { SetProperty(ref _thresholding, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float DynamicThresholdingRatio
        {
            get { return _dynamicThresholdingRatio; }
            set { SetProperty(ref _dynamicThresholdingRatio, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public VarianceType? VarianceType
        {
            get { return _varianceType; }
            set { SetProperty(ref _varianceType, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool UseKarrasSigmas
        {
            get { return _useKarrasSigmas; }
            set { SetProperty(ref _useKarrasSigmas, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool UseBetaSigmas
        {
            get { return _useBetaSigmas; }
            set { SetProperty(ref _useBetaSigmas, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool UseExponentialSigmas
        {
            get { return _useExponentialSigmas; }
            set { SetProperty(ref _useExponentialSigmas, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool UseFlowSigmas
        {
            get { return _useFlowSigmas; }
            set { SetProperty(ref _useFlowSigmas, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float SigmaMin
        {
            get { return _sigmaMin; }
            set { SetProperty(ref _sigmaMin, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float SigmaMax
        {
            get { return _sigmaMax; }
            set { SetProperty(ref _sigmaMax, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public FinalSigmasType FinalSigmasType
        {
            get { return _finalSigmasType; }
            set { SetProperty(ref _finalSigmasType, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public InterpolationType InterpolationType
        {
            get { return _interpolationType; }
            set { SetProperty(ref _interpolationType, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public TimestepType TimestepType
        {
            get { return _timestepType; }
            set { SetProperty(ref _timestepType, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool RescaleBetasZeroSNR
        {
            get { return _rescaleBetasZeroSNR; }
            set { SetProperty(ref _rescaleBetasZeroSNR, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool SetAlphaToOne
        {
            get { return _setAlphaToOne; }
            set { SetProperty(ref _setAlphaToOne, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float TimestepScaling
        {
            get { return _timestepScaling; }
            set { SetProperty(ref _timestepScaling, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float Shift
        {
            get { return _shift; }
            set { SetProperty(ref _shift, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float BaseShift
        {
            get { return _baseShift; }
            set { SetProperty(ref _baseShift, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float MaxShift
        {
            get { return _maxShift; }
            set { SetProperty(ref _maxShift, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float ShiftTerminal
        {
            get { return _shiftTerminal; }
            set { SetProperty(ref _shiftTerminal, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool UseDynamicShifting
        {
            get { return _useDynamicShifting; }
            set { SetProperty(ref _useDynamicShifting, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float FlowShift
        {
            get { return _flowShift; }
            set { SetProperty(ref _flowShift, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float SNRShiftScale
        {
            get { return _sNRShiftScale; }
            set { SetProperty(ref _sNRShiftScale, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public TimeShiftType TimeShiftType
        {
            get { return _timeShiftType; }
            set { SetProperty(ref _timeShiftType, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float Rho
        {
            get { return _rho; }
            set { SetProperty(ref _rho, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public int SolverOrder
        {
            get { return _solverOrder; }
            set { SetProperty(ref _solverOrder, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public SolverType SolverType
        {
            get { return _solverType; }
            set { SetProperty(ref _solverType, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public AlgorithmType AlgorithmType
        {
            get { return _algorithmType; }
            set { SetProperty(ref _algorithmType, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool LowerOrderFinal
        {
            get { return _lowerOrderFinal; }
            set { SetProperty(ref _lowerOrderFinal, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool StochasticSampling
        {
            get { return _stochasticSampling; }
            set { SetProperty(ref _stochasticSampling, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float Eta
        {
            get { return _eta; }
            set { SetProperty(ref _eta, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float SNoise
        {
            get { return _sNoise; }
            set { SetProperty(ref _sNoise, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool InvertSigmas
        {
            get { return _invertSigmas; }
            set { SetProperty(ref _invertSigmas, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool SkipPrkSteps
        {
            get { return _skipPrkSteps; }
            set { SetProperty(ref _skipPrkSteps, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool PredictX0
        {
            get { return _predictX0; }
            set { SetProperty(ref _predictX0, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool EulerAtFinal
        {
            get { return _eulerAtFinal; }
            set { SetProperty(ref _eulerAtFinal, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool UseLuLambdas
        {
            get { return _useLuLambdas; }
            set { SetProperty(ref _useLuLambdas, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public int NoiseSamplerSeed
        {
            get { return _noiseSamplerSeed; }
            set { SetProperty(ref _noiseSamplerSeed, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float SigmaData
        {
            get { return _sigmaData; }
            set { SetProperty(ref _sigmaData, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public SigmaScheduleType SigmaScheduleType
        {
            get { return _sigmaScheduleType; }
            set { SetProperty(ref _sigmaScheduleType, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public UpscaleModeType UpscaleMode
        {
            get { return _upscaleMode; }
            set { SetProperty(ref _upscaleMode, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public int Stages
        {
            get { return _stages; }
            set { SetProperty(ref _stages, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float Gamma
        {
            get { return _gamma; }
            set { SetProperty(ref _gamma, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public int PredictorOrder
        {
            get { return _predictorOrder; }
            set { SetProperty(ref _predictorOrder, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public int CorrectorOrder
        {
            get { return _correctorOrder; }
            set { SetProperty(ref _correctorOrder, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public List<float> ScaleFactors
        {
            get { return _scaleFactors; }
            set { SetProperty(ref _scaleFactors, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public List<float> StageRange
        {
            get { return _stageRange; }
            set { SetProperty(ref _stageRange, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public List<int> DisableCorrector
        {
            get { return _disableCorrector; }
            set { SetProperty(ref _disableCorrector, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float SValue
        {
            get { return _sValue; }
            set { SetProperty(ref _sValue, value); }
        }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public float Scaler
        {
            get { return _scaler; }
            set { SetProperty(ref _scaler, value); }
        }


        public SchedulerOptions ToClientOptions()
        {
            return new SchedulerOptions
            {
                Scheduler = Scheduler,
                NumTrainTimesteps = NumTrainTimesteps,
                Eta = Eta,
                SNoise = SNoise,
                AlgorithmType = AlgorithmType,
                BaseImageSeqLen = BaseImageSeqLen,
                BaseShift = BaseShift,
                BetaEnd = BetaEnd,
                BetaSchedule = BetaSchedule,
                BetaStart = BetaStart,
                ClipSample = ClipSample,
                ClipSampleRange = ClipSampleRange,
                CorrectorOrder = CorrectorOrder,
                DisableCorrector = DisableCorrector,
                DynamicThresholdingRatio = DynamicThresholdingRatio,
                EulerAtFinal = EulerAtFinal,
                FinalSigmasType = FinalSigmasType,
                FlowShift = FlowShift,
                Gamma = Gamma,
                InterpolationType = InterpolationType,
                InvertSigmas = InvertSigmas,
                LowerOrderFinal = LowerOrderFinal,
                MaxImageSeqLen = MaxImageSeqLen,
                MaxShift = MaxShift,
                NoiseSamplerSeed = NoiseSamplerSeed == 0 ? null : NoiseSamplerSeed,
                OriginalInferenceSteps = OriginalInferenceSteps,
                PredictionType = PredictionType,
                PredictorOrder = PredictorOrder,
                PredictX0 = PredictX0,
                RescaleBetasZeroSNR = RescaleBetasZeroSNR,
                Rho = Rho,
                SampleMaxValue = SampleMaxValue,
                ScaleFactors = ScaleFactors,
                Scaler = Scaler,
                SetAlphaToOne = SetAlphaToOne,
                Shift = Shift,
                ShiftTerminal = ShiftTerminal == 0 ? null : ShiftTerminal,
                SigmaData = SigmaData,
                SigmaMax = SigmaMax == 0 ? null : SigmaMax,
                SigmaMin = SigmaMin == 0 ? null : SigmaMin,
                SigmaScheduleType = SigmaScheduleType,
                SkipPrkSteps = SkipPrkSteps,
                SNRShiftScale = SNRShiftScale,
                SolverOrder = SolverOrder,
                SolverType = SolverType,
                StageRange = StageRange,
                Stages = Stages,
                StepsOffset = StepsOffset,
                StochasticSampling = StochasticSampling,
                SValue = SValue,
                Thresholding = Thresholding,
                TimeShiftType = TimeShiftType,
                TimestepScaling = TimestepScaling,
                TimestepSpacing = TimestepSpacing,
                TimestepType = TimestepType,
                UpscaleMode = UpscaleMode,
                UseBetaSigmas = UseBetaSigmas,
                UseDynamicShifting = UseDynamicShifting,
                UseExponentialSigmas = UseExponentialSigmas,
                UseFlowSigmas = UseFlowSigmas,
                UseKarrasSigmas = UseKarrasSigmas,
                UseLuLambdas = UseLuLambdas,
                VarianceType = VarianceType,
            };
        }

        public override string ToString()
        {
            return _scheduler.GetDisplayName();
        }

        public bool Equals(SchedulerInputOptions other) => ReferenceEquals(this, other);
        public override int GetHashCode() => RuntimeHelpers.GetHashCode(this);
    }
}

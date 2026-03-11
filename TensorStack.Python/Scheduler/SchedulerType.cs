using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace TensorStack.Python.Scheduler
{
    public enum SchedulerType
    {
        // LMSDiscreteScheduler
        [Display(Name = "LMS")]
        LMS = 0,

        // EulerDiscreteScheduler
        [Display(Name = "Euler")]
        Euler = 1,

        // EulerAncestralDiscreteScheduler
        [Display(Name = "Euler Ancestral")]
        EulerAncestral = 2,

        // DDPMScheduler
        [Display(Name = "DDPM")]
        DDPM = 3,

        // DDIMScheduler
        [Display(Name = "DDIM")]
        DDIM = 4,

        // KDPM2DiscreteScheduler
        [Display(Name = "KDPM2")]
        KDPM2 = 5,

        // KDPM2AncestralDiscreteScheduler
        [Display(Name = "KDPM2-Ancestral")]
        KDPM2Ancestral = 6,

        // DDPMWuerstchenScheduler
        [Display(Name = "DDPM-Wuerstchen")]
        DDPMWuerstchen = 10,

        // LCMScheduler
        [Display(Name = "LCM")]
        LCM = 20,

        // FlowMatchEulerDiscreteScheduler
        [Display(Name = "FlowMatch-Euler")]
        FlowMatchEuler = 30,

        // FlowMatchHeunDiscreteScheduler
        [Display(Name = "FlowMatch-Heun")]
        FlowMatchHeun = 31,

        // PNDMScheduler
        [Display(Name = "PNDM")]
        PNDM = 40,

        // HeunDiscreteScheduler
        [Display(Name = "Heun")]
        Heun = 41,

        // UniPCMultistepScheduler
        [Display(Name = "UniPC Multistep")]
        UniPCMultistep = 42,

        // DPMSolverMultistepScheduler
        [Display(Name = "DPM Solver Multistep")]
        DPMSolverMultistep = 43,

        // DPMSolverSinglestepScheduler
        [Display(Name = "DPM Single Step")]
        DPMSolverSinglestep = 45,

        // DPMSolverSDEScheduler
        [Display(Name = "DPM Solver SDE")]
        DPMSolverSDE = 46,

        // DEISMultistepScheduler
        [Display(Name = "DEIS Multistep")]
        DEISMultistep = 47,

        // EDMEulerScheduler
        [Display(Name = "EDM Euler")]
        EDMEuler = 48,

        // EDMDPMSolverMultistepScheduler
        [Display(Name = "EDM DPM Solver Multistep")]
        EDMDPMSolverMultistep = 49,

        // FlowMatchLCMScheduler
        [Display(Name = "FlowMatch-LCM")]
        FlowMatchLCM = 50,

        // IPNDMScheduler
        [Display(Name = "IPNDM")]
        IPNDM = 51,

        // CogVideoXDDIMScheduler
        [Display(Name = "CogVideoX DDIM")]
        CogVideoXDDIM = 52,

        // CogVideoXDPMScheduler
        [Display(Name = "CogVideoX DPM")]
        CogVideoXDPM = 53,

        // HeliosScheduler
        [Display(Name = "Helios")]
        Helios = 54,

        // HeliosDMDScheduler
        [Display(Name = "Helios DMD")]
        HeliosDMD = 55,

        // TCDScheduler
        [Display(Name = "TCD")]
        TCD = 56,

        // SCMScheduler
        [Display(Name = "SCM")]
        SCM = 57,

        // SASolverScheduler
        [Display(Name = "SA Solver")]
        SASolver = 58,

        // LTXEulerAncestralRFScheduler
        [Display(Name = "LTX Euler Ancestral")]
        LTXEulerAncestral = 59,
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
        DPMSolverPlus = 1,

        [JsonStringEnumMemberName("sde-dpmsolver")]
        SDE_DPMSolver = 2,

        [JsonStringEnumMemberName("sde-dpmsolver++")]
        SDE_DPMSolverPlus = 3,

        [JsonStringEnumMemberName("deis")]
        DEIS = 4,

        [JsonStringEnumMemberName("data_prediction")]
        DataPrediction = 5,

        [JsonStringEnumMemberName("noise_prediction")]
        NoisePrediction = 6
    }

    public enum SolverType
    {

        [JsonStringEnumMemberName("midpoint")]
        Midpoint = 0,

        [JsonStringEnumMemberName("heun")]
        Heun = 1,

        [JsonStringEnumMemberName("bh1")]
        BH1 = 2,

        [JsonStringEnumMemberName("bh2")]
        BH2 = 3,

        [JsonStringEnumMemberName("logrho")]
        LogRho = 4
    }

    public enum BetaScheduleType
    {
        [JsonStringEnumMemberName("linear")]
        Linear = 0,

        [JsonStringEnumMemberName("scaled_linear")]
        ScaledLinear = 1,

        [JsonStringEnumMemberName("cosine")]
        Cosine = 2,

        [JsonStringEnumMemberName("squaredcos_cap_v2")]
        SquaredCosine = 3,

        [JsonStringEnumMemberName("sigmoid")]
        Sigmoid = 4,

        [JsonStringEnumMemberName("laplace")]
        Laplace = 5,

        [JsonStringEnumMemberName("exp")]
        Exponential = 6
    }

    public enum PredictionType
    {
        [JsonStringEnumMemberName("epsilon")]
        Epsilon = 0,

        [JsonStringEnumMemberName("v_prediction")]
        Variable = 1,

        [JsonStringEnumMemberName("sample")]
        Sample = 2,

        [JsonStringEnumMemberName("flow_prediction")]
        FlowPrediction = 3,

        [JsonStringEnumMemberName("trigflow")]
        Trigflow = 4
    }

    public enum VarianceType
    {
        [JsonStringEnumMemberName("fixed_small")]
        FixedSmall = 0,

        [JsonStringEnumMemberName("fixed_small_log")]
        FixedSmallLog = 1,

        [JsonStringEnumMemberName("fixed_large")]
        FixedLarge = 2,

        [JsonStringEnumMemberName("fixed_large_log")]
        FixedLargeLog = 3,

        [JsonStringEnumMemberName("learned")]
        Learned = 4,

        [JsonStringEnumMemberName("learned_range")]
        LearnedRange = 5
    }

    public enum TimeShiftType
    {
        [JsonStringEnumMemberName("linear")]
        Linear = 0,

        [JsonStringEnumMemberName("exponential")]
        Exponential = 1
    }

    public enum AlphaTransformType
    {
        [JsonStringEnumMemberName("cosine")]
        Cosine = 0,

        [JsonStringEnumMemberName("exp")]
        Exponential = 1,

        [JsonStringEnumMemberName("laplace")]
        Laplace = 2
    }

    public enum FinalSigmasType
    {
        [JsonStringEnumMemberName("zero")]
        Zero = 0,

        [JsonStringEnumMemberName("sigma_min")]
        SigmaMin = 1
    }

    public enum InterpolationType
    {
        [JsonStringEnumMemberName("linear")]
        Linear = 0,

        [JsonStringEnumMemberName("log_linear")]
        LogLinear = 1
    }

    public enum TimestepType
    {
        [JsonStringEnumMemberName("discrete")]
        Discrete = 0,

        [JsonStringEnumMemberName("continuous")]
        Continuous = 1
    }

    public enum SigmaScheduleType
    {
        [JsonStringEnumMemberName("karras")]
        Karras = 0,

        [JsonStringEnumMemberName("exponential")]
        Exponential = 1
    }

    public enum UpscaleModeType
    {
        [JsonStringEnumMemberName("nearest")]
        Nearest = 0,

        [JsonStringEnumMemberName("linear")]
        Linear = 1,

        [JsonStringEnumMemberName("bilinear")]
        Bilinear = 2,

        [JsonStringEnumMemberName("bicubic")]
        Bicubic = 3,

        [JsonStringEnumMemberName("trilinear")]
        Trilinear = 4,

        [JsonStringEnumMemberName("area")]
        Area = 5,

        [JsonStringEnumMemberName("nearest-exact")]
        NearestExact = 6
    }
}

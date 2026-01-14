using System.ComponentModel.DataAnnotations;

namespace TensorStack.Python.Common
{
    public enum SchedulerType
    {
        // LCMScheduler
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
        [Display(Name = "FlowMatch-EulerDiscrete")]
        FlowMatchEulerDiscrete = 30,

        // FlowMatchHeunDiscreteScheduler
        [Display(Name = "FlowMatch-HeunDiscrete")]
        FlowMatchHeunDiscrete = 31,

        // PNDMScheduler
        [Display(Name = "PNDM")]
        PNDM = 40,

        // HeunDiscreteScheduler
        [Display(Name = "Heun")]
        Heun = 41,

        // UniPCMultistepScheduler
        [Display(Name = "UniPC")]
        UniPC = 42,

        // DPMSolverMultistepScheduler
        [Display(Name = "DPM Multi Step")]
        DPMM = 43,

        // DPMSolverMultistepInverseScheduler
        [Display(Name = "DPM Multi Step Inverse")]
        DPMMInverse = 44,

        // DPMSolverSinglestepScheduler
        [Display(Name = "DPM Single Step")]
        DPMS = 45,

        // DPMSolverSDEScheduler
        [Display(Name = "DPM SDE")]
        DPMSDE = 46,

        // DEISMultistepScheduler
        [Display(Name = "DEIS Multistep")]
        DEISM = 47,

        // EDMEulerScheduler
        [Display(Name = "EDM Euler")]
        EDM = 48,

        // EDMDPMSolverMultistepScheduler
        [Display(Name = "EDM DPMSolver Multi Step")]
        EDMM = 49,


        // FlowMatchLCMScheduler
        [Display(Name = "FlowMatch-LCM")]
        FlowMatchLCM = 50,

        // IPNDMScheduler
        [Display(Name = "IPNDM")]
        IPNDM = 51
    }


    public static class SchedulerExtensions
    {
        public static bool IsKarras(this SchedulerType scheduler)
        {
            return scheduler switch
            {
                SchedulerType.DDIM => true,
                SchedulerType.DDPM => true,
                SchedulerType.PNDM => true,
                SchedulerType.LMS => true,
                SchedulerType.Euler => true,
                SchedulerType.Heun => true,
                SchedulerType.EulerAncestral => true,
                SchedulerType.DPMM => true,
                SchedulerType.DPMS => true,
                SchedulerType.KDPM2 => true,
                SchedulerType.KDPM2Ancestral => true,
                SchedulerType.DEISM => true,
                SchedulerType.UniPC => true,
                SchedulerType.DPMSDE => true,
                SchedulerType.EDM => true,
                _ => false
            };
        }


        public static bool IsMultiStep(this SchedulerType scheduler)
        {
            return scheduler switch
            {
                SchedulerType.DEISM => true,
                SchedulerType.DPMS => true,
                SchedulerType.DPMM => true,
                SchedulerType.DPMMInverse => true,
                SchedulerType.EDMM => true,
                SchedulerType.UniPC => true,
                _ => false
            };
        }


        public static bool IsFlowMatch(this SchedulerType scheduler)
        {
            return scheduler switch
            {
                SchedulerType.FlowMatchLCM => true,
                SchedulerType.FlowMatchEulerDiscrete => true,
                SchedulerType.FlowMatchHeunDiscrete => true,
                _ => false
            };
        }


        public static bool IsClipSample(this SchedulerType scheduler)
        {
            return scheduler switch
            {
                SchedulerType.DDIM => true,
                SchedulerType.DDPM => true,
                SchedulerType.Heun => true,
                SchedulerType.LCM => true,
                _ => false
            };
        }


        public static bool IsThreshold(this SchedulerType scheduler)
        {
            return scheduler switch
            {
                SchedulerType.DDIM => true,
                SchedulerType.DDPM => true,
                SchedulerType.DEISM => true,
                SchedulerType.DPMM => true,
                SchedulerType.DPMMInverse => true,
                SchedulerType.EDMM => true,
                SchedulerType.LCM => true,
                SchedulerType.UniPC => true,
                _ => false
            };
        }


        public static bool IsTimestep(this SchedulerType scheduler)
        {
            return scheduler switch
            {
                SchedulerType.DDIM => true,
                SchedulerType.DDPM => true,
                SchedulerType.DEISM => true,
                SchedulerType.DPMS => true,
                SchedulerType.DPMM => true,
                SchedulerType.DPMMInverse => true,
                SchedulerType.DPMSDE => true,
                SchedulerType.Euler => true,
                SchedulerType.EulerAncestral => true,
                SchedulerType.FlowMatchHeunDiscrete => true,
                SchedulerType.Heun => true,
                SchedulerType.KDPM2 => true,
                SchedulerType.KDPM2Ancestral => true,
                SchedulerType.LCM => true,
                SchedulerType.LMS => true,
                SchedulerType.PNDM => true,
                SchedulerType.UniPC => true,
                _ => false
            };
        }


        public static bool IsStochastic(this SchedulerType scheduler)
        {
            return scheduler switch
            {
                SchedulerType.EDM => true,
                SchedulerType.Euler => true,
                SchedulerType.FlowMatchEulerDiscrete => true,
                SchedulerType.FlowMatchHeunDiscrete => true,
                _ => false
            };
        }

    }
}

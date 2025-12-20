using System.ComponentModel.DataAnnotations;

namespace TensorStack.Python.Common
{
    public enum SchedulerType
    {
        [Display(Name = "LMS")]
        LMS = 0,
        [Display(Name = "Euler")]
        Euler = 1,
        [Display(Name = "Euler Ancestral")]
        EulerAncestral = 2,
        [Display(Name = "DDPM")]
        DDPM = 3,
        [Display(Name = "DDIM")]
        DDIM = 4,
        [Display(Name = "KDPM2")]
        KDPM2 = 5,
        [Display(Name = "KDPM2-Ancestral")]
        KDPM2Ancestral = 6,
        [Display(Name = "DDPM-Wuerstchen")]
        DDPMWuerstchen = 10,
        [Display(Name = "LCM")]
        LCM = 20,
        [Display(Name = "FlowMatch-EulerDiscrete")]
        FlowMatchEulerDiscrete = 30,
        [Display(Name = "FlowMatch-HeunDiscrete")]
        FlowMatchHeunDiscrete = 31,
        [Display(Name = "PNDM")]
        PNDM = 40,
        [Display(Name = "Heun")]
        Heun = 41,
        [Display(Name = "UniPC")]
        UniPC = 42,
        [Display(Name = "DPM Multi Step")]
        DPMM = 43,
        [Display(Name = "DPM Single Step")]
        DPMS = 44,
        [Display(Name = "DPM SDE")]
        DPMSDE = 45,
    }
}

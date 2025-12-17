using System.ComponentModel.DataAnnotations;

namespace TensorStack.Python.Options
{
    public record PythonOptions
    {
        public int Seed { get; set; }
        public string Prompt { get; set; }
        public string NegativePrompt { get; set; }
        public float GuidanceScale { get; set; }
        public int Steps { get; set; }
        public int Height { get; set; }
        public int Width { get; set; }
        public int Frames { get; set; }
        public float OutputFrameRate { get; set; }
        public float Shift { get; set; }
        public float FlowShift { get; set; }
        public SchedulerType Scheduler { get; set; }
    }

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

        PNDM = 40,
        Heun = 41,
        UniPC = 42,
        DPMM = 43,
        DPMS = 44,
    }
}

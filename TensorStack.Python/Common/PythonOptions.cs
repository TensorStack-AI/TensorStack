using System.Collections.Generic;

namespace TensorStack.Python.Common
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
        public float Strength { get; set; }
        public SchedulerType Scheduler { get; set; }
        public SchedulerType[] Schedulers { get; set; }
        public List<LoraOptions> LoraOptions { get; set; }
    }
}

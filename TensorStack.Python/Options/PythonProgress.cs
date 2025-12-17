using TensorStack.Common.Pipeline;

namespace TensorStack.Python.Options
{
    public record PythonProgress : IRunProgress
    {
        public string Process { get; set; }
        public string Message { get; set; }
        public int Iteration { get; set; }
        public int Iterations { get; set; }
        public float IterationsPerSecond { get; set; }
        public float SecondsPerIteration { get; set; }
        public float Downloaded { get; set; }
        public float DownloadTotal { get; set; }
        public float DownloadSpeed { get; set; }
    }
}

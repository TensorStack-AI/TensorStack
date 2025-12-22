using System;
using TensorStack.Common.Pipeline;

namespace TensorStack.Python.Common
{
    public record PipelineProgress : IRunProgress
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
        public bool IsLoading => Process == "Loading";
        public bool IsGenerating => Process == "Generate" && Iterations > 0;
        public bool IsDownloading => DownloadTotal > 0 || Downloaded > 0 || DownloadSpeed > 0;

        public readonly static IProgress<PipelineProgress> ConsoleCallback = new Progress<PipelineProgress>(Console.WriteLine);
    }
}

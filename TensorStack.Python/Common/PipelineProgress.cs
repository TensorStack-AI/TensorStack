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
        public string DownloadModel { get; set; }
        public string DownloadFile { get; set; }
        public bool IsLoading => Process == "Load";
        public bool IsGenerating => Process == "Generate" && Iterations > 0;
        public bool IsDownloading => Process == "Download";

        public readonly static IProgress<PipelineProgress> ConsoleCallback = new Progress<PipelineProgress>(Console.WriteLine);
    }
}

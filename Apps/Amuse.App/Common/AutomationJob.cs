using System.IO;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TensorStack.Audio;
using TensorStack.Common;
using TensorStack.Image;
using TensorStack.Video;

namespace Amuse.App.Common
{
    public sealed record AutomationJob
    {
        public int Id { get; init; }
        public int Count { get; init; }
        public DiffusionInputOptions DiffusionOptions { get; init; }
        public UpscaleInputOptions UpscaleOptions { get; init; }
        public ExtractInputOptions ExtractOptions { get; init; }
        public InterpolateInputOptions InterpolateOptions { get; init; }

        public string OutputFile { get; init; }
        public ImageInput[] InputImages { get; init; }
        public TextInput[] InputTexts { get; init; }
        public VideoInputStream[] VideoStreams { get; init; }
        public AudioInputStream[] AudioStreams { get; init; }

        public async Task SaveAsync(ImageInput imageInput)
        {
            if (string.IsNullOrWhiteSpace(OutputFile))
                return;

            await imageInput.SaveAsync(OutputFile);
        }

        public async Task SaveAsync(TextInput textInput)
        {
            if (string.IsNullOrWhiteSpace(OutputFile))
                return;

            await textInput.SaveAsync(OutputFile);
        }

        public Task SaveAsync(VideoInputStream videoInput)
        {
            if (string.IsNullOrWhiteSpace(OutputFile))
                return Task.CompletedTask;

            File.Copy(videoInput.SourceFile, OutputFile, true);
            return Task.CompletedTask;
        }


        public Task SaveAsync(AudioInputStream audioInput)
        {
            if (string.IsNullOrWhiteSpace(OutputFile))
                return Task.CompletedTask;

            File.Copy(audioInput.SourceFile, OutputFile, true);
            return Task.CompletedTask;
        }

        public bool Equals(AutomationJob other) => ReferenceEquals(this, other);
        public override int GetHashCode() => RuntimeHelpers.GetHashCode(this);
    }
}
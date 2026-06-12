namespace TensorStack.WPF
{
    public interface IUIConfiguration
    {
        string DirectoryTemp { get; }
        double VolumeInput { get; set; }
        double VolumeOutput { get; set; }
        bool IsVolumeInputMute { get; set; }
        bool IsVolumeOutputMute { get; set; }
    }


    public record DefaultUIConfiguration : IUIConfiguration
    {
        public string DirectoryTemp { get; init; }
        public double VolumeInput { get; set; } = 0.1;
        public double VolumeOutput { get; set; } = 0.1;
        public bool IsVolumeInputMute { get; set; }
        public bool IsVolumeOutputMute { get; set; }
    }
}
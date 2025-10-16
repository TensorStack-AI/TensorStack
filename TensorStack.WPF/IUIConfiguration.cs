namespace TensorStack.WPF
{
    public interface IUIConfiguration
    {
        string DirectoryTemp { get; }
    }


    public record DefaultUIConfiguration : IUIConfiguration
    {
        public string DirectoryTemp { get; init; }
    }
}

